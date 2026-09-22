"""Real vs mock embedding backend + OpenSearch knn dim sync (P4).

``RCR_EMBED_BACKEND``:
  - ``mock`` / ``hash`` — char-hash vectors (unit tests only)
  - ``local`` / ``semantic`` — local semantic vectors (hashed n-grams, or
    optional ``fastembed`` if installed) — **use this when CAIIS has chat only**
  - ``litellm`` / ``remote`` / ``real`` — LiteLLM / CAIIS ``/embeddings``
  - ``auto`` (default) — remote when a *real* embed model is configured;
    otherwise ``local`` (not char-hash mock)

Chat-only models (e.g. Nemotron) are **not** treated as embed endpoints.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

log = logging.getLogger("rcr_router.embeddings")

DEFAULT_LOCAL_DIM = 384
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Models that expose /chat/completions but typically not /embeddings
_CHAT_ONLY_MARKERS = (
    "nemotron",
    "gpt-oss",
    "chatgpt",
    "claude",
    "command-r",
    "mixtral",
    "llama-3",
    "llama3",
    "super-120b",
    "instruct",
)
_EMBED_MARKERS = (
    "embed",
    "embedqa",
    "e5-",
    "bge-",
    "minilm",
    "gte-",
    "text-embedding",
    "nomic",
    "snowflake-arctic-embed",
    "nv-embed",
)


@dataclass
class EmbedStatus:
    backend: str  # mock | local | litellm
    dim: int
    model: str = ""
    api_base: str = ""
    ok: bool = True
    detail: str = ""
    index_dim: int | None = None
    dim_mismatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def looks_like_chat_only_model(model: str | None) -> bool:
    """True when ``model`` is almost certainly chat-only (no /embeddings)."""
    m = (model or "").lower().strip()
    if not m:
        return False
    # Strip provider prefixes
    for prefix in ("openai/", "hosted_vllm/", "nvidia_nim/", "nvidia/"):
        if m.startswith(prefix):
            m = m[len(prefix) :]
    if any(tok in m for tok in _EMBED_MARKERS):
        return False
    return any(tok in m for tok in _CHAT_ONLY_MARKERS)


def embed_backend_preference() -> str:
    raw = os.getenv("RCR_EMBED_BACKEND", "auto").strip().lower()
    if raw in {"mock", "hash"}:
        return "mock"
    if raw in {"local", "semantic", "ngram", "fastembed"}:
        return "local"
    if raw in {"litellm", "remote", "real"}:
        return "litellm"
    return "auto"


def embed_fallback_mock_allowed() -> bool:
    return os.getenv("RCR_EMBED_FALLBACK_MOCK", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def local_embed_dim() -> int:
    raw = os.getenv("RCR_LOCAL_EMBED_DIM", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    env = env_embed_dim()
    if env and env not in {8}:  # 8 is the old mock default — prefer 384 for local
        return env
    return DEFAULT_LOCAL_DIM


def remote_embed_configured(
    *,
    embed_model: str | None = None,
    embed_api_base: str | None = None,
    api_key: str | None = None,
) -> bool:
    """True when we have enough config to attempt a live /embeddings call."""
    model = (
        embed_model
        or os.getenv("CLOUDERA_AI_INFERENCE_EMBED_MODEL")
        or os.getenv("CAIIS_EMBED_MODEL")
        or os.getenv("LITELLM_EMBED_MODEL")
        or ""
    ).strip()
    if not model or looks_like_chat_only_model(model):
        return False
    base = (
        embed_api_base
        or os.getenv("CLOUDERA_AI_INFERENCE_EMBED_API_BASE")
        or os.getenv("CAIIS_EMBED_API_BASE")
        or os.getenv("CLOUDERA_AI_INFERENCE_API_BASE")
        or os.getenv("LITELLM_API_BASE")
        or os.getenv("OPENAI_API_BASE")
        or ""
    ).strip()
    from .cloudera import coerce_api_key

    key = coerce_api_key(api_key)
    if key and (
        base
        or model.startswith(("text-embedding", "openai/text-embedding"))
        or any(tok in model.lower() for tok in _EMBED_MARKERS)
    ):
        return True
    return bool(model and base and key)


def resolve_embed_backend(
    *,
    embed_model: str | None = None,
    embed_api_base: str | None = None,
    api_key: str | None = None,
) -> str:
    pref = embed_backend_preference()
    if pref in {"mock", "local", "litellm"}:
        return pref
    # auto: real remote embed → litellm; else local semantic (not char-hash mock)
    if remote_embed_configured(
        embed_model=embed_model, embed_api_base=embed_api_base, api_key=api_key
    ):
        return "litellm"
    return "local"


def env_embed_dim() -> int | None:
    raw = os.getenv("OPENSEARCH_EMBED_DIM", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return None


def _l2_normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def _hashed_ngram_embed(texts: list[str], *, dim: int) -> list[list[float]]:
    """Dependency-free semantic-ish vectors via signed hashed unigrams+bigrams."""
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        toks = _TOKEN_RE.findall((text or "").lower())
        grams = list(toks)
        grams.extend(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        if not grams:
            grams = ["_empty_"]
        for g in grams:
            digest = hashlib.md5(g.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            # mild IDF-ish: rarer longer tokens weigh a bit more
            weight = 1.0 + min(2.0, len(g) / 12.0)
            vec[idx] += sign * weight
        out.append(_l2_normalize(vec))
    return out


@lru_cache(maxsize=1)
def _fastembed_model():
    from fastembed import TextEmbedding

    name = os.getenv("RCR_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    return TextEmbedding(model_name=name)


def local_semantic_embed(texts: list[str], *, dim: int | None = None) -> tuple[list[list[float]], str]:
    """Return (vectors, backend_label). Prefer fastembed when installed."""
    dim = dim or local_embed_dim()
    prefer = os.getenv("RCR_LOCAL_EMBED_IMPL", "auto").strip().lower()
    if prefer in {"fastembed", "auto"}:
        try:
            model = _fastembed_model()
            vectors = [list(map(float, v)) for v in model.embed(list(texts))]
            if vectors and vectors[0]:
                return vectors, "local:fastembed"
        except Exception as exc:  # noqa: BLE001
            if prefer == "fastembed":
                raise
            log.info("fastembed unavailable (%s); using hashed n-grams", type(exc).__name__)
    return _hashed_ngram_embed(texts, dim=dim), "local:ngram"


def index_embedding_dim(client, index: str) -> int | None:
    try:
        if not client.indices.exists(index=index):
            return None
        mapping = client.indices.get_mapping(index=index)
        props = mapping[index]["mappings"].get("properties") or {}
        emb = props.get("embedding") or {}
        dim = emb.get("dimension")
        return int(dim) if dim is not None else None
    except Exception:
        return None


def probe_embeddings(llm: Any, *, text: str = "rcr embed probe") -> EmbedStatus:
    """Run one embed call and report backend + dimension."""
    model = getattr(llm, "embed_model", None) or ""
    base = getattr(llm, "embed_api_base", None) or getattr(llm, "api_base", None) or ""
    chat_only = looks_like_chat_only_model(str(model))
    try:
        try:
            vectors = llm.embed([text], input_type="query")
        except TypeError:
            vectors = llm.embed([text])
        if not vectors or not vectors[0]:
            return EmbedStatus(
                backend=getattr(llm, "last_embed_backend", "unknown") or "unknown",
                dim=0,
                model=str(model or ""),
                api_base=str(base or ""),
                ok=False,
                detail="empty vector",
            )
        dim = len(vectors[0])
        backend = getattr(llm, "last_embed_backend", None) or "unknown"
        detail = ""
        if backend == "mock":
            detail = "using mock/hash embeddings (tests only)"
        elif str(backend).startswith("local"):
            detail = "local semantic embeddings (no CAIIS /embeddings required)"
        if chat_only and backend == "mock":
            detail = (
                "chat-only model configured as embed "
                f"({model}); set RCR_EMBED_BACKEND=local or a real embed endpoint"
            )
        return EmbedStatus(
            backend=str(backend),
            dim=dim,
            model=str(model or ""),
            api_base=str(base or ""),
            ok=True,
            detail=detail,
        )
    except Exception as exc:  # noqa: BLE001
        return EmbedStatus(
            backend="error",
            dim=0,
            model=str(model or ""),
            api_base=str(base or ""),
            ok=False,
            detail=f"{type(exc).__name__}: {exc}"[:300],
        )


def sync_opensearch_dim(
    embed_dim: int,
    *,
    recreate_on_mismatch: bool | None = None,
    client=None,
) -> EmbedStatus:
    """Ensure working + knowledge indices exist at ``embed_dim``."""
    from .opensearch_store import (
        KNOWLEDGE_INDEX,
        WORKING_INDEX,
        ensure_indices,
        opensearch_available,
        opensearch_client_from_env,
    )

    if recreate_on_mismatch is None:
        recreate_on_mismatch = os.getenv(
            "RCR_RECREATE_INDICES_ON_DIM_MISMATCH", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}

    if not opensearch_available(client):
        os.environ["OPENSEARCH_EMBED_DIM"] = str(embed_dim)
        return EmbedStatus(
            backend="n/a",
            dim=embed_dim,
            ok=True,
            detail="opensearch unavailable; dim env updated only",
        )

    client = client or opensearch_client_from_env()
    dims = {
        WORKING_INDEX: index_embedding_dim(client, WORKING_INDEX),
        KNOWLEDGE_INDEX: index_embedding_dim(client, KNOWLEDGE_INDEX),
    }
    existing = next((d for d in dims.values() if d is not None), None)
    mismatch = existing is not None and existing != embed_dim

    ensure_indices(
        client,
        embed_dim=embed_dim,
        recreate_on_mismatch=recreate_on_mismatch and mismatch,
    )
    os.environ["OPENSEARCH_EMBED_DIM"] = str(embed_dim)

    new_dims = {
        WORKING_INDEX: index_embedding_dim(client, WORKING_INDEX),
        KNOWLEDGE_INDEX: index_embedding_dim(client, KNOWLEDGE_INDEX),
    }
    still_mismatch = any(d is not None and d != embed_dim for d in new_dims.values())
    detail = ""
    if mismatch and not recreate_on_mismatch:
        detail = (
            f"index dim={existing} != embed dim={embed_dim}; "
            "set RCR_RECREATE_INDICES_ON_DIM_MISMATCH=true or recreate from Settings"
        )
        log.warning(detail)
    elif mismatch and recreate_on_mismatch:
        detail = f"recreated indices for dim {embed_dim} (was {existing})"
    else:
        detail = f"indices ready at dim={embed_dim}"

    return EmbedStatus(
        backend="opensearch",
        dim=embed_dim,
        index_dim=new_dims.get(KNOWLEDGE_INDEX) or new_dims.get(WORKING_INDEX),
        dim_mismatch=still_mismatch,
        ok=not still_mismatch,
        detail=detail,
    )


def prepare_embeddings(llm: Any, *, recreate_on_mismatch: bool | None = None) -> EmbedStatus:
    """Probe llm.embed, set OPENSEARCH_EMBED_DIM, sync indices."""
    status = probe_embeddings(llm)
    if not status.ok or status.dim <= 0:
        return status
    os.environ["OPENSEARCH_EMBED_DIM"] = str(status.dim)

    sync = sync_opensearch_dim(
        status.dim, recreate_on_mismatch=recreate_on_mismatch
    )
    status.index_dim = sync.index_dim
    status.dim_mismatch = sync.dim_mismatch
    if sync.detail:
        status.detail = (status.detail + "; " if status.detail else "") + sync.detail
    if not sync.ok:
        status.ok = False
    return status
