"""Token length helpers for RCR budgets (paper-aligned accounting).

Prefer tiktoken / LiteLLM counters; fall back to a char heuristic — never use
raw whitespace word counts for budget fill when a real counter is available.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

TokenBackend = Literal["tiktoken", "litellm", "approx"]

_BACKEND: TokenBackend | None = None


def token_backend() -> TokenBackend:
    """Force backend via RCR_TOKEN_BACKEND=tiktoken|litellm|approx, else auto."""
    global _BACKEND
    forced = os.getenv("RCR_TOKEN_BACKEND", "").strip().lower()
    if forced in {"tiktoken", "litellm", "approx"}:
        return forced  # type: ignore[return-value]
    if _BACKEND is not None:
        return _BACKEND
    try:
        import tiktoken  # noqa: F401

        _BACKEND = "tiktoken"
    except Exception:
        try:
            import litellm  # noqa: F401

            _BACKEND = "litellm"
        except Exception:
            _BACKEND = "approx"
    return _BACKEND


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    import tiktoken

    name = os.getenv("RCR_TIKTOKEN_ENCODING", "cl100k_base")
    return tiktoken.get_encoding(name)


def count_tokens(text: str, *, model: str | None = None) -> int:
    """Return approximate LLM token count for ``text`` (≥ 1 if non-empty)."""
    if not text:
        return 0
    backend = token_backend()
    if backend == "tiktoken":
        try:
            return max(1, len(_tiktoken_encoder().encode(text)))
        except Exception:
            pass
    if backend in {"tiktoken", "litellm"}:
        try:
            import litellm

            model_name = model or os.getenv("RCR_TOKEN_MODEL", "gpt-4o-mini")
            n = litellm.token_counter(model=model_name, text=text)
            return max(1, int(n))
        except Exception:
            pass
    # ~4 chars / token — closer to BPE than whitespace words for English
    return max(1, (len(text) + 3) // 4)


def count_tokens_many(texts: list[str], *, model: str | None = None) -> list[int]:
    return [count_tokens(t, model=model) for t in texts]
