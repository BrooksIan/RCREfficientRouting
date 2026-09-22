from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from .cloudera import coerce_api_key, ClouderaLLMSettings, resolve_cloudera_settings
from .model_routing import as_openai_compatible_model


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "mock"


@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    traces: list[dict] = field(default_factory=list)

    def add(self, resp: LLMResponse, *, role: str, round_idx: int) -> None:
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.calls += 1
        self.traces.append(
            {
                "role": role,
                "round": round_idx,
                "model": resp.model,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "text_preview": resp.text[:200],
            }
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> LLMResponse: ...

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        input_type: str | None = None,
    ) -> list[list[float]]: ...


class MockLLMClient:
    """Deterministic client for unit tests / offline demos (no network)."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or int(os.getenv("OPENSEARCH_EMBED_DIM", "8"))
        self.last_embed_backend = "mock"

    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> LLMResponse:
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        ul = user.lower()
        sl = system.lower()
        if "expert judge" in sl or (
            "json object" in ul and "justification" in ul and "user query" in ul
        ):
            text = (
                '{"score": 4, "justification": '
                '"Relevant and mostly complete mock answer."}'
            )
        elif "exactly 3 search intents" in ul or "json array of strings" in ul:
            text = '["Acme Q3 revenue earnings", "cloud subscription contribution", "hardware vs cloud mix"]'
        elif "extract only factual claims" in ul:
            text = (
                "- Acme Corp reported 18% YoY revenue growth in Q3\n"
                "- Growth primarily driven by cloud subscription services\n"
                "- Hardware sales were flat"
            )
        elif "answer in one sentence" in ul:
            text = "Acme Corp Q3 revenue growth was driven primarily by cloud subscription services."
        elif "routing controller" in sl or "json array of agent roles" in ul:
            text = '["planner","searcher","recommender"]'
        elif "classify user prompts" in sl or "allowed labels:" in ul:
            # Task classifier mock — prefer synthesis for Acme-style queries
            label = "synthesis"
            if any(k in ul for k in ("code", "bug", "implement", "refactor")):
                label = "coding"
            elif any(k in ul for k in ("extract", "evidence", "passage")):
                label = "extraction"
            elif any(k in ul for k in ("route", "which agent", "classify this query")):
                label = "routing"
            elif any(k in ul for k in ("why", "decompose", "strategy", "reason")):
                label = "reasoning"
            text = f'{{"task":"{label}","confidence":0.82,"rationale":"mock"}}'
        elif "planner" in sl:
            text = f"PLAN: decompose query into subquestions about: {user[:120]}"
        elif "searcher" in sl or "evidence extraction" in sl:
            text = f"EVIDENCE: retrieved facts relevant to: {user[:120]}"
        elif "recommender" in sl or "synthesis quality" in sl:
            text = f"ANSWER: based on plan and evidence for: {user[:120]}"
        else:
            text = f"OK: {user[:120]}"
        return LLMResponse(
            text=text,
            prompt_tokens=max(1, len(user.split())),
            completion_tokens=max(1, len(text.split())),
            model=model or "mock-llm",
        )

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        input_type: str | None = None,
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for i, ch in enumerate(text.lower()):
                vec[i % self.dim] += (ord(ch) % 31) / 31.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        self.last_embed_backend = "mock"
        return vectors


class LiteLLMClient:
    """LiteLLM wrapper with Cloudera AI Inference / OpenAI-compatible support."""

    def __init__(
        self,
        chat_model: str | None = None,
        embed_model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        embed_api_base: str | None = None,
        settings: ClouderaLLMSettings | None = None,
        drop_params: bool = True,
    ) -> None:
        self.settings = settings
        if settings is not None:
            self.chat_model = chat_model or settings.litellm_chat_model
            self.embed_model = embed_model or settings.litellm_embed_model
            self.api_base = api_base or settings.api_base
            self.api_key = api_key or settings.api_key
            self.embed_api_base = embed_api_base or settings.embed_api_base or self.api_base
        else:
            self.chat_model = chat_model or os.getenv("LITELLM_MODEL", "gpt-4o-mini")
            self.embed_model = embed_model or os.getenv(
                "LITELLM_EMBED_MODEL", "text-embedding-3-small"
            )
            self.api_base = api_base or os.getenv("LITELLM_API_BASE") or os.getenv(
                "OPENAI_API_BASE"
            )
            self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv(
                "OPENAI_API_KEY"
            )
            self.embed_api_base = embed_api_base or self.api_base
        self.drop_params = drop_params
        self.last_embed_backend = "unset"

    def _apply_litellm_globals(self) -> None:
        import litellm

        if self.drop_params:
            litellm.drop_params = True
        # Hide repeated "Give Feedback / Get Help" footers on handled errors
        litellm.suppress_debug_info = True

    def _chat_kwargs(
        self,
        model_name: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model_name}
        base = api_base if api_base is not None else self.api_base
        key = coerce_api_key(
            api_key if api_key is not None else self.api_key,
            fallback=self.api_key,
        )
        if base:
            kwargs["api_base"] = base
        if key:
            kwargs["api_key"] = key
        return kwargs

    def _embed_kwargs(
        self,
        model_name: str,
        *,
        input_type: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model_name}
        base = self.embed_api_base or self.api_base
        key = coerce_api_key(self.api_key)
        if base:
            kwargs["api_base"] = base
        if key:
            kwargs["api_key"] = key
        # NVIDIA EmbedQA (asymmetric) requires input_type=query|passage
        itype = input_type or os.getenv("RCR_EMBED_INPUT_TYPE", "").strip() or None
        if not itype:
            mid = (model_name or "").lower()
            if "embedqa" in mid or "nv-embedqa" in mid or "nv-embed" in mid:
                itype = "query"
        if itype:
            kwargs["extra_body"] = {"input_type": itype}
        return kwargs

    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> LLMResponse:
        import litellm

        self._apply_litellm_globals()
        model_name = model or self.chat_model
        # OpenAI-compatible api_base (CAIIS / NIM): always openai/<served-id>
        effective_base = api_base if api_base is not None else self.api_base
        if effective_base:
            model_name = as_openai_compatible_model(model_name or "")
        elif (
            self.settings is not None
            and not self.settings.via_proxy
            and model_name
        ):
            model_name = as_openai_compatible_model(model_name)

        try:
            resp = litellm.completion(
                messages=messages,
                **self._chat_kwargs(model_name, api_base=api_base, api_key=api_key),
            )
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            msg = str(exc)
            if "Authentication" in name or "401" in msg or "Authentication failed" in msg:
                raise RuntimeError(
                    "Cloudera AI Inference authentication failed (401). "
                    "In Settings, set API key to a CDP JWT (starts with eyJ…), not the endpoint URL. "
                    "Or export CDP_TOKEN in .env and clear the bad key. "
                    f"model={model_name} base={effective_base}"
                ) from exc
            raise
        choice = resp.choices[0].message.content or ""
        choice = self.clean_completion_text(choice)
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=choice,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=model_name,
        )

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        input_type: str | None = None,
    ) -> list[list[float]]:
        import litellm

        from .embeddings import (
            embed_fallback_mock_allowed,
            local_semantic_embed,
            looks_like_chat_only_model,
            resolve_embed_backend,
        )

        self._apply_litellm_globals()
        model_name = model or self.embed_model
        backend = resolve_embed_backend(
            embed_model=model_name,
            embed_api_base=self.embed_api_base or self.api_base,
            api_key=self.api_key,
        )

        if backend == "mock":
            out = MockLLMClient().embed(texts)
            self.last_embed_backend = "mock"
            return out

        # Chat-only CAIIS (Nemotron, etc.) or explicit local backend
        if backend == "local" or looks_like_chat_only_model(model_name) or not model_name:
            out, label = local_semantic_embed(texts)
            self.last_embed_backend = label
            if out and out[0]:
                os.environ["OPENSEARCH_EMBED_DIM"] = str(len(out[0]))
            return out

        if (
            self.settings is not None
            and not self.settings.via_proxy
            and model_name
        ):
            model_name = as_openai_compatible_model(model_name)
        elif self.embed_api_base or self.api_base:
            model_name = as_openai_compatible_model(model_name or "")

        try:
            resp = litellm.embedding(
                input=texts,
                **self._embed_kwargs(model_name, input_type=input_type),
            )
            data = resp.data if hasattr(resp, "data") else resp["data"]
            out = []
            for row in data:
                emb = row["embedding"] if isinstance(row, dict) else row.embedding
                out.append(list(emb))
            self.last_embed_backend = "litellm"
            if out and out[0]:
                os.environ["OPENSEARCH_EMBED_DIM"] = str(len(out[0]))
            return out
        except Exception as exc:  # noqa: BLE001
            if embed_fallback_mock_allowed():
                import logging

                logging.getLogger("rcr_router.llm").warning(
                    "Embedding failed (%s); falling back to local semantic embeddings. "
                    "Set RCR_EMBED_BACKEND=local or a dedicated embed endpoint. detail=%s",
                    type(exc).__name__,
                    str(exc)[:200],
                )
                out, label = local_semantic_embed(texts)
                self.last_embed_backend = label
                if out and out[0]:
                    os.environ["OPENSEARCH_EMBED_DIM"] = str(len(out[0]))
                return out
            self.last_embed_backend = "error"
            raise

    @staticmethod
    def clean_completion_text(text: str) -> str:
        """Strip reasoning-model wrappers and Nemotron chain-of-thought dumps."""
        import re

        cleaned = text or ""
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", cleaned, flags=re.IGNORECASE)

        # Prefer text after explicit final-answer anchors (last match wins).
        anchors = list(
            re.finditer(
                r"(?is)(?:^|\n)\s*(?:"
                r"the\s+prime\s+numbers\b[^\n]*:\s*|"
                r"(?:thus\s+)?final\s+answer\s*[:\-–]?\s*|"
                r"thus\s+(?:the\s+)?answer\s*[:\-–]?\s*|"
                r"provide\s+only\s+(?:the\s+)?final\s+answer\b[^\n]*\n+|"
                r"answer\s*:\s*"
                r")",
                cleaned,
            )
        )
        if anchors:
            tail = cleaned[anchors[-1].end() :].strip()
            # Drop trailing "list of …" meta lines before the real list
            tail = re.sub(
                r"(?is)^(?:list of[^\n]*\n+)+",
                "",
                tail,
            ).strip()
            if len(tail) >= 20:
                cleaned = tail

        # Drop private scratch / self-check lines common in Nemotron dumps.
        drop_line = re.compile(
            r"(?i)^(?:\s*)(?:"
            r"check(?:ing)?(?:\s+prime)?\b.*"
            r"|let'?s\s+(?:compute|list|verify|generate|check)\b.*"
            r"|i'?ll\s+(?:generate|produce|list|compute|verify)\b.*"
            r"|we\s+(?:need|can|must)\b.*"
            r"|start\s*:\s*\d+.*"
            r"|should\s+not\s+only\s+be\b.*"
            r"|make\s+sure\b.*"
            r"|also\s+(?:check|need|include)\b.*"
            r"|did\s+we\s+miss\b.*"
            r"|up\s+to\s+\d+\s*:\s*.*"
            r"|primes?\s+up\s+to\b.*"
            r"|segment\s+\d+.*"
            r"|compile\s+full\s+list.*"
            r"|now\s+produce\b.*"
            r"|all\s+good\.?"
            r"|proceed\.?"
            r"|will\s+present\b.*"
            r"|provide\s+(?:answer|only)\b.*"
            r"|forbidden:.*"
            r"|citations?\s+like\b.*"
            r"|good\.?"
            r")\s*$"
        )
        lines = [ln for ln in cleaned.splitlines() if not drop_line.match(ln)]
        cleaned = "\n".join(lines).strip()

        cleaned = re.sub(
            r"(?i)(?:\s*check(?:\s+prime)?\s+\d+\??\s*yes\.?\s*){3,}",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        # Prefer the densest comma-separated number list (typical final prime list).
        list_blocks = re.findall(
            r"(?:\d+\s*,\s*){8,}\d+",
            cleaned,
        )
        if list_blocks:
            best = max(list_blocks, key=lambda s: (s.count(","), len(s)))
            # Only replace when the list is a clear final payload
            if best.count(",") >= 8 and (
                len(cleaned) > len(best) * 1.15 or "check" in cleaned.lower()
                or "we " in cleaned.lower()[:80]
            ):
                cleaned = best

        # If still huge, keep densest numeric paragraph.
        if len(cleaned) > 1200:
            paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
            scored: list[tuple[int, str]] = []
            for p in paras:
                nums = len(re.findall(r"\b\d+\b", p))
                scored.append((nums, p))
            if scored:
                scored.sort(key=lambda x: (x[0], len(x[1])))
                best_p = scored[-1][1]
                if scored[-1][0] >= 10:
                    cleaned = best_p

        return cleaned.strip()


def build_llm_client(force_mock: bool | None = None) -> LLMClient:
    """Prefer Cloudera/LiteLLM when configured; otherwise MockLLMClient."""
    if force_mock is None:
        force_mock = os.getenv("RCR_USE_MOCK_LLM", "").lower() in {"1", "true", "yes"}
    if force_mock:
        return MockLLMClient()

    settings = resolve_cloudera_settings()
    if settings is not None:
        return LiteLLMClient(settings=settings)

    has_key = bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("AZURE_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("LITELLM_API_KEY")
    )
    has_base = bool(os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_API_BASE"))
    if has_key or has_base:
        return LiteLLMClient()
    return MockLLMClient()


def role_model_map() -> dict[str, str]:
    """Backward-compatible helper: role → LiteLLM model string."""
    from .model_routing import active_model_registry

    return active_model_registry().role_model_map()
