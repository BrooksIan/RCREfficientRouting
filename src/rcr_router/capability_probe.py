"""Probe user-supplied LLMs to discover role strengths and assign agents."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Iterable

from .endpoint_store import (
    CapabilityScores,
    EndpointStoreData,
    UserEndpoint,
    save_store,
)
from .llm import LLMClient, LiteLLMClient, MockLLMClient
from .model_routing import ModelEndpoint, as_openai_compatible_model
from .cloudera import coerce_api_key


# Short, role-specific probes. Scoring is heuristic + instruction-following checks.
PROBES: dict[str, dict[str, str]] = {
    "planner": {
        "system": "You are evaluating planning skill. Follow instructions exactly.",
        "user": (
            "Decompose this question into exactly 3 search intents as a JSON array of strings. "
            "No prose.\nQuestion: What drove Acme Corp revenue growth in Q3?"
        ),
    },
    "searcher": {
        "system": "You are evaluating evidence extraction. Follow instructions exactly.",
        "user": (
            "Passage:\n"
            "Acme Corp reported 18% YoY revenue growth in Q3, primarily driven by cloud "
            "subscription services. Hardware sales were flat.\n\n"
            "Extract ONLY factual claims as a markdown bullet list. No commentary."
        ),
    },
    "recommender": {
        "system": "You are evaluating synthesis quality. Follow instructions exactly.",
        "user": (
            "Facts:\n"
            "- Acme Q3 revenue +18% YoY\n"
            "- Growth driven by cloud subscriptions\n"
            "- Hardware sales flat\n\n"
            "Answer in ONE sentence: What drove Acme Corp Q3 revenue growth?"
        ),
    },
    "router": {
        "system": "You are a routing controller. Output JSON only.",
        "user": (
            "Return a JSON array of agent roles to run for this query, choosing from "
            '["planner","searcher","recommender"]. Query: "Summarize the evidence for '
            'Acme cloud growth."'
        ),
    },
}


def prior_from_model_id(model_id: str) -> dict[str, float]:
    """Cheap prior from the model id / name before live probing."""
    mid = (model_id or "").lower()
    scores = {"planner": 0.45, "searcher": 0.45, "recommender": 0.45, "router": 0.45}
    if any(k in mid for k in ("nemotron", "reason", "o1", "r1", "thinking")):
        scores["planner"] += 0.2
        scores["recommender"] += 0.1
    if any(k in mid for k in ("extract", "embed", "mini", "small", "flash", "haiku")):
        scores["searcher"] += 0.15
        scores["router"] += 0.15
    if any(k in mid for k in ("gpt-4", "sonnet", "opus", "large", "70b", "120b", "super")):
        scores["recommender"] += 0.15
        scores["planner"] += 0.1
    if any(k in mid for k in ("router", "gate", "classify")):
        scores["router"] += 0.25
    if any(k in mid for k in ("code", "coder", "codellama", "starcoder", "deepseek-coder")):
        scores["planner"] += 0.1
        scores["searcher"] += 0.1
    return scores


# Operator-declared intended task → soft boosts for agent roles.
TASK_ROLE_PRIORS: dict[str, dict[str, float]] = {
    "generic": {},
    "coding": {"planner": 0.12, "searcher": 0.1, "recommender": 0.05},
    "reasoning": {"planner": 0.2, "recommender": 0.1},
    "extraction": {"searcher": 0.2, "router": 0.05},
    "synthesis": {"recommender": 0.2, "planner": 0.05},
    "routing": {"router": 0.25, "searcher": 0.05},
}


def prior_from_intended_task(intended_task: str) -> dict[str, float]:
    task = (intended_task or "generic").strip().lower()
    boosts = TASK_ROLE_PRIORS.get(task, {})
    base = {"planner": 0.0, "searcher": 0.0, "recommender": 0.0, "router": 0.0}
    for role, delta in boosts.items():
        base[role] = delta
    return base


def combined_prior(model_id: str, intended_task: str = "generic") -> dict[str, float]:
    """Blend model-id prior with operator-declared intended task."""
    mid = prior_from_model_id(model_id)
    task = prior_from_intended_task(intended_task)
    return {
        role: min(1.0, mid[role] + task.get(role, 0.0))
        for role in ("planner", "searcher", "recommender", "router")
    }

def _score_planner(text: str) -> tuple[float, str]:
    text = text.strip()
    arr = re.search(r"\[[^\]]*\]", text, flags=re.S)
    if not arr:
        return 0.2, "no JSON array"
    items = re.findall(r'"([^"]+)"', arr.group(0))
    if len(items) == 3:
        return 0.95, "exactly 3 intents"
    if 2 <= len(items) <= 4:
        return 0.7, f"{len(items)} intents"
    return 0.35, "weak structure"


def _score_searcher(text: str) -> tuple[float, str]:
    bullets = [ln for ln in text.splitlines() if re.match(r"^\s*[-*•]", ln)]
    hits = sum(
        1
        for kw in ("18%", "cloud", "hardware", "subscription", "flat", "revenue")
        if kw.lower() in text.lower()
    )
    score = min(1.0, 0.25 * len(bullets) + 0.12 * hits)
    return score, f"bullets={len(bullets)} fact_hits={hits}"


def _score_recommender(text: str) -> tuple[float, str]:
    sentences = [s for s in re.split(r"[.!?]\s+", text.strip()) if s.strip()]
    one = 1.0 if len(sentences) <= 2 and 8 <= len(text.split()) <= 60 else 0.5
    hits = sum(1 for kw in ("cloud", "subscription", "18", "growth") if kw in text.lower())
    score = min(1.0, 0.4 * one + 0.2 * hits)
    return score, f"sentences~{len(sentences)} hits={hits}"


def _score_router(text: str) -> tuple[float, str]:
    arr = re.search(r"\[[^\]]*\]", text)
    if not arr:
        return 0.15, "no JSON array"
    roles = {r.lower() for r in re.findall(r'"?(planner|searcher|recommender)"?', arr.group(0), flags=re.I)}
    if roles == {"searcher", "recommender"} or roles == {"planner", "searcher", "recommender"}:
        return 0.95, f"roles={sorted(roles)}"
    if roles:
        return 0.55, f"roles={sorted(roles)}"
    return 0.2, "empty roles"


SCORERS = {
    "planner": _score_planner,
    "searcher": _score_searcher,
    "recommender": _score_recommender,
    "router": _score_router,
}


def _client_for_endpoint(ep: UserEndpoint) -> LLMClient:
    model = as_openai_compatible_model(ep.model_id)
    return LiteLLMClient(
        chat_model=model,
        embed_model=None,
        api_base=ep.api_base.rstrip("/"),
        api_key=coerce_api_key(ep.api_key),
    )


def probe_endpoint(
    ep: UserEndpoint,
    *,
    llm: LLMClient | None = None,
    roles: Iterable[str] = ("planner", "searcher", "recommender", "router"),
) -> CapabilityScores:
    """Run live probes (or injected client) and combine with model-id + task prior."""
    prior = combined_prior(ep.model_id, ep.normalized_task())
    client = llm or _client_for_endpoint(ep)
    scores = CapabilityScores()
    latencies: list[float] = []

    for role in roles:
        probe = PROBES[role]
        started = time.time()
        try:
            resp = client.complete(
                [
                    {"role": "system", "content": probe["system"]},
                    {"role": "user", "content": probe["user"]},
                ],
                model=None if isinstance(client, MockLLMClient) else as_openai_compatible_model(
                    ep.model_id
                ),
                api_base=ep.api_base or None,
                api_key=coerce_api_key(ep.api_key),
            )
            text = resp.text or ""
            elapsed = time.time() - started
            latencies.append(elapsed)
            raw_score, note = SCORERS[role](text)
            # Blend prior (30%) with live probe (70%)
            final = 0.3 * prior[role] + 0.7 * raw_score
            setattr(scores, role, round(final, 3))
            scores.notes[role] = f"{note}; task={ep.normalized_task()}"
            scores.samples[role] = text[:400]
        except Exception as exc:  # noqa: BLE001
            setattr(scores, role, round(0.2 * prior[role], 3))
            scores.notes[role] = f"probe_failed: {exc}; task={ep.normalized_task()}"
            scores.samples[role] = ""
            latencies.append(time.time() - started)

    scores.latency_sec = round(sum(latencies) / max(1, len(latencies)), 3)
    # Slight router bonus for lower latency
    if scores.latency_sec > 0:
        latency_bonus = max(0.0, min(0.1, (5.0 - scores.latency_sec) / 50.0))
        scores.router = round(min(1.0, scores.router + latency_bonus), 3)
    return scores


def assign_roles(store: EndpointStoreData) -> dict[str, dict]:
    """Greedy unique assignment: best remaining model per role priority."""
    configured = store.configured_endpoints()
    if not configured:
        return {}

    # role priority: recommender/planner matter most, then searcher, then router
    order = ("recommender", "planner", "searcher", "router")
    used_slots: set[int] = set()
    assignments: dict[str, dict] = {}

    for role in order:
        ranked: list[tuple[float, UserEndpoint]] = []
        for ep in configured:
            cap = store.capabilities.get(ep.model_id) or store.capabilities.get(str(ep.slot))
            if not cap:
                continue
            ranked.append((float(cap.get(role, 0.0)), ep))
        ranked.sort(key=lambda x: x[0], reverse=True)
        chosen = None
        for score, ep in ranked:
            if ep.slot not in used_slots:
                chosen = (score, ep)
                break
        if chosen is None and ranked:
            # allow reuse if fewer than needed models
            score, ep = ranked[0]
            chosen = (score, ep)
        if chosen is None:
            continue
        score, ep = chosen
        used_slots.add(ep.slot)
        assignments[role] = {
            "slot": ep.slot,
            "label": ep.label or ep.model_id,
            "model_id": ep.model_id,
            "api_base": ep.api_base,
            "api_key": coerce_api_key(ep.api_key) or ep.api_key,
            "intended_task": ep.normalized_task(),
            "score": score,
        }
    return assignments

def run_discovery(store: EndpointStoreData, *, use_mock: bool = False) -> EndpointStoreData:
    """Probe all configured endpoints and write role assignments."""
    capabilities: dict[str, dict] = {}
    for ep in store.configured_endpoints():
        llm = MockLLMClient() if use_mock else None
        scores = probe_endpoint(ep, llm=llm)
        capabilities[ep.model_id] = scores.as_dict()
        capabilities[str(ep.slot)] = scores.as_dict()
    store.capabilities = capabilities
    store.role_assignments = assign_roles(store)
    store.last_probed_at = datetime.now(timezone.utc).isoformat()
    save_store(store)
    return store


def assignments_to_model_endpoints(store: EndpointStoreData) -> dict[str, ModelEndpoint]:
    out: dict[str, ModelEndpoint] = {}
    for role, info in (store.role_assignments or {}).items():
        model = as_openai_compatible_model(info.get("model_id") or "")
        out[role] = ModelEndpoint(
            model=model,
            api_base=info.get("api_base"),
            api_key=coerce_api_key(info.get("api_key")),
            label=info.get("label") or info.get("model_id") or role,
        )
    return out
