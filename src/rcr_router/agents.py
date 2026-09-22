from __future__ import annotations

import re

from .llm import LLMClient, LLMResponse, UsageTracker, LiteLLMClient
from .model_routing import ModelRegistry, active_model_registry
from .models import AgentRole, MemoryItem


SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.PLANNER: (
        "You are the Planner agent. Decompose the user query into a short plan "
        "and search intents. Use only the provided context. Be concise. "
        "Output only the plan — no chain-of-thought narration."
    ),
    AgentRole.SEARCHER: (
        "You are the Searcher agent. Extract and state evidence facts from the "
        "provided context that answer the plan/search intents. Quote concrete facts "
        "(ingredients, amounts, steps, numbers). Ignore context about unrelated topics. "
        "Never reply with only citation numbers like [1] [2]. "
        "Output only the evidence — no working notes or self-checks."
    ),
    AgentRole.RECOMMENDER: (
        "You are the Recommender agent. Output ONLY the final answer the user asked for. "
        "Include concrete details from context (lists, amounts, steps, numbers). "
        "Do NOT include chain-of-thought, planning, self-checks ('Check prime 2? yes'), "
        "or meta commentary about citations. "
        "Citations [n] are optional and must never be the whole answer. "
        "No special citation markup."
    ),
}

# Nemotron / tool-style citation chrome that leaks into answers
_CITATION_CHROME = re.compile(
    r"【\s*\d+\s*[†‡][^】]*】|"  # 【5†L1-L5】
    r"\[\^\d+\]"
)
_BRACKET_CITE = re.compile(r"\[\s*\d+\s*\]")
_NEMOTRON_CITE = re.compile(r"【\s*(\d+)\s*[†‡][^】]*】")
_ONLY_CITATIONS = re.compile(
    r"^(?:\s*(?:\[\s*\d+\s*\]|【\s*\d+\s*[†‡][^】]*】)\s*)+$",
    re.UNICODE,
)


def clean_agent_text(text: str) -> str:
    # First strip CoT / think blocks, then citation chrome.
    cleaned = LiteLLMClient.clean_completion_text(text or "")
    cleaned = _CITATION_CHROME.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned).strip()
    return cleaned


def is_citation_only(text: str) -> bool:
    """True when the model answered with only [n] / 【n†…】 refs and no substance."""
    raw = (text or "").strip()
    if not raw:
        return True
    if _ONLY_CITATIONS.match(raw):
        return True
    t = clean_agent_text(raw)
    if not t:
        return True  # was only citation chrome
    if _ONLY_CITATIONS.match(t):
        return True
    residual = _BRACKET_CITE.sub(" ", t)
    residual = _NEMOTRON_CITE.sub(" ", residual)
    residual = re.sub(r"\s+", " ", residual).strip(" .;,-")
    words = [w for w in residual.split() if w]
    if len(words) < 4 and (_BRACKET_CITE.search(raw) or _NEMOTRON_CITE.search(raw)):
        return True
    return False


def resolve_citations_to_context(
    text: str,
    context_items: list[MemoryItem],
) -> str:
    """If answer is mostly [n] / 【n†…】 refs, expand those indices into context snippets."""
    raw = text or ""
    nums = [int(n) for n in re.findall(r"\[\s*(\d+)\s*\]", raw)]
    nums.extend(int(n) for n in _NEMOTRON_CITE.findall(raw))
    if not nums or not context_items:
        return ""
    parts: list[str] = []
    seen: set[int] = set()
    for n in nums:
        if n in seen or n < 1 or n > len(context_items):
            continue
        seen.add(n)
        snippet = (context_items[n - 1].text or "").strip()
        if snippet and not snippet.upper().startswith("USER_QUERY:"):
            parts.append(snippet)
    return "\n\n".join(parts)


def synthesize_answer_from_context(context_items: list[MemoryItem], query: str) -> str:
    """Deterministic fallback when the LLM returns empty / citation-only."""
    evidence = [
        (i.text or "").strip()
        for i in context_items
        if (i.text or "").strip()
        and not (i.text or "").lstrip().upper().startswith("USER_QUERY:")
    ]
    if not evidence:
        return ""
    # Prefer denser evidence bullets
    evidence.sort(key=lambda t: len(t.split()), reverse=True)
    bullets = evidence[:4]
    header = f"Based on the routed evidence for: {query.strip()}"
    return header + "\n\n" + "\n\n".join(f"- {b}" for b in bullets)


def format_context(items: list[MemoryItem]) -> str:
    """Compact prompt context — content only, no duplicated struct/YAML chrome."""
    if not items:
        return "(no routed context)"
    parts = []
    for i, item in enumerate(items, 1):
        body = (item.text or "").strip()
        # Legacy YAML blobs: prefer the inner text: field if present
        if body.startswith("kind:") and "\ntext:" in body:
            try:
                import yaml

                parsed = yaml.safe_load(body)
                if isinstance(parsed, dict) and parsed.get("text"):
                    body = str(parsed["text"]).strip()
            except Exception:
                pass
        kind = item.struct_kind or ""
        if kind == "plan_step" and not re.match(r"^\d+\.\s", body):
            step = (item.structure or {}).get("step")
            prefix = f"plan {step}: " if step is not None else "plan: "
            parts.append(f"[{i}] {prefix}{body}")
        elif kind in {"fact", "evidence", "answer"}:
            parts.append(f"[{i}] {kind}: {body}")
        else:
            parts.append(f"[{i}] {body}")
    return "\n\n".join(parts)


class AgentRunner:
    def __init__(
        self,
        llm: LLMClient,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry or active_model_registry()

    def run(
        self,
        *,
        role: AgentRole,
        query: str,
        context_items: list[MemoryItem],
        usage: UsageTracker | None = None,
        round_idx: int = 0,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        extra_instructions: str = "",
    ) -> LLMResponse:
        context = format_context(context_items)
        user_content = (
            f"User query:\n{query}\n\nRouted context:\n{context}\n"
            f"{extra_instructions}"
        ).strip()
        if role == AgentRole.RECOMMENDER:
            user_content += (
                "\n\nReminder: reply with ONLY the final answer content "
                "(facts/list/steps). No reasoning, no 'Check …? yes' lines, "
                "no citation-only replies."
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[role]},
            {"role": "user", "content": user_content},
        ]
        endpoint = self.registry.for_role(role)
        chosen_model = model or endpoint.model
        chosen_base = api_base if api_base is not None else endpoint.api_base
        chosen_key = api_key if api_key is not None else endpoint.api_key
        try:
            resp = self.llm.complete(
                messages,
                model=chosen_model,
                api_base=chosen_base,
                api_key=chosen_key,
            )
        except Exception as exc:  # noqa: BLE001
            resp = LLMResponse(
                text=f"[agent error · {role.value}] {exc}",
                prompt_tokens=0,
                completion_tokens=0,
                model=chosen_model or "error",
            )
        text = clean_agent_text(resp.text)
        # Repair citation-only / empty recommender (and searcher) outputs.
        if role in {AgentRole.RECOMMENDER, AgentRole.SEARCHER} and is_citation_only(text):
            expanded = resolve_citations_to_context(resp.text, context_items)
            if expanded:
                text = expanded
            elif role == AgentRole.RECOMMENDER:
                text = synthesize_answer_from_context(context_items, query) or text
        resp = LLMResponse(
            text=text,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            model=resp.model,
        )
        if usage is not None:
            usage.add(resp, role=role.value, round_idx=round_idx)
        return resp
