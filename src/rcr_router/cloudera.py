"""Cloudera AI Inference / CML auth helpers for LiteLLM."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClouderaLLMSettings:
    """Resolved settings for OpenAI-compatible Cloudera endpoints."""

    api_base: str
    api_key: str
    chat_model: str
    embed_model: str | None
    embed_api_base: str | None
    via_proxy: bool

    @property
    def litellm_chat_model(self) -> str:
        # openai/ prefix routes through OpenAI-compatible provider in LiteLLM
        if self.via_proxy:
            return self.chat_model  # proxy alias, e.g. cloudera-chat
        if self.chat_model.startswith(("openai/", "hosted_vllm/", "nvidia_nim/")):
            return self.chat_model
        return f"openai/{self.chat_model}"

    @property
    def litellm_embed_model(self) -> str | None:
        if not self.embed_model:
            return None
        if self.via_proxy:
            return self.embed_model
        if self.embed_model.startswith(("openai/", "hosted_vllm/", "nvidia_nim/")):
            return self.embed_model
        return f"openai/{self.embed_model}"


def _read_cml_jwt(path: str = "/tmp/jwt") -> str | None:
    """CML Workbench sessions often expose a JWT at /tmp/jwt."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        if raw.startswith("{"):
            data = json.loads(raw)
            return data.get("access_token") or data.get("token")
        return raw
    except Exception:
        return None


def resolve_api_key() -> str | None:
    """Prefer explicit keys, then CDP_TOKEN, then CML /tmp/jwt."""
    for env_name in (
        "CLOUDERA_AI_INFERENCE_API_KEY",
        "CDP_TOKEN",
        "LITELLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        val = os.getenv(env_name)
        if val:
            return val.strip()
    return _read_cml_jwt(os.getenv("CML_JWT_PATH", "/tmp/jwt"))


def is_plausible_api_key(key: str | None) -> bool:
    """Reject empty values and accidental URL pastes into the API key field."""
    if not key or not str(key).strip():
        return False
    k = str(key).strip()
    if k.startswith(("http://", "https://")):
        return False
    if "://" in k:
        return False
    # CDP JWTs are long; proxy master keys are short but not URLs
    return len(k) >= 8


def coerce_api_key(key: str | None, *, fallback: str | None = None) -> str | None:
    """Return key if usable, else fallback, else resolve_api_key()."""
    if is_plausible_api_key(key):
        return str(key).strip()
    if is_plausible_api_key(fallback):
        return str(fallback).strip()
    return resolve_api_key()


def resolve_cloudera_settings() -> ClouderaLLMSettings | None:
    """Build Cloudera/LiteLLM settings when a Cloudera or proxy base URL is set.

    Direct CAIIS (Cloudera AI Inference Service):
      CLOUDERA_AI_INFERENCE_API_BASE=https://.../endpoints/<name>/v1
      CLOUDERA_AI_INFERENCE_MODEL=<registry model name>
      CDP_TOKEN=<workload auth token>

    Via LiteLLM proxy (recommended for AMP):
      LITELLM_API_BASE=http://litellm:4000
      LITELLM_MODEL=cloudera-chat
      LITELLM_API_KEY=sk-rcr-local  (proxy master key)
    """
    proxy_base = (os.getenv("LITELLM_API_BASE") or "").rstrip("/")
    cai_base = (
        os.getenv("CLOUDERA_AI_INFERENCE_API_BASE")
        or os.getenv("CAIIS_API_BASE")
        or os.getenv("OPENAI_API_BASE")
        or ""
    ).rstrip("/")

    api_key = resolve_api_key()
    if not api_key:
        return None

    # Proxy mode: app talks only to LiteLLM; proxy holds Cloudera creds
    if proxy_base and not os.getenv("RCR_FORCE_DIRECT_CAIIS"):
        chat = os.getenv("LITELLM_MODEL", "cloudera-chat")
        embed = os.getenv("LITELLM_EMBED_MODEL", "cloudera-embed")
        return ClouderaLLMSettings(
            api_base=proxy_base,
            api_key=api_key,
            chat_model=chat,
            embed_model=embed,
            embed_api_base=proxy_base,
            via_proxy=True,
        )

    if not cai_base:
        return None

    chat = (
        os.getenv("CLOUDERA_AI_INFERENCE_MODEL")
        or os.getenv("CAIIS_MODEL")
        or os.getenv("LITELLM_MODEL")
        or "llm"
    )
    embed = (
        os.getenv("CLOUDERA_AI_INFERENCE_EMBED_MODEL")
        or os.getenv("CAIIS_EMBED_MODEL")
        or os.getenv("LITELLM_EMBED_MODEL")
    )
    # Don't treat chat-only CAIIS models (e.g. Nemotron) as embed endpoints —
    # that forces failed /embeddings calls and mock fallbacks.
    from .embeddings import looks_like_chat_only_model

    if embed and looks_like_chat_only_model(embed):
        embed = None
    embed_base = (
        os.getenv("CLOUDERA_AI_INFERENCE_EMBED_API_BASE")
        or os.getenv("CAIIS_EMBED_API_BASE")
        or cai_base
    ).rstrip("/")

    return ClouderaLLMSettings(
        api_base=cai_base,
        api_key=api_key,
        chat_model=chat,
        embed_model=embed,
        embed_api_base=embed_base if embed else None,
        via_proxy=False,
    )


def cloudera_configured() -> bool:
    return resolve_cloudera_settings() is not None
