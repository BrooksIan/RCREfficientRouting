"""Multi-LLM role routing: map agent roles to LiteLLM/Cloudera endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .cloudera import coerce_api_key, resolve_api_key, resolve_cloudera_settings
from .models import AgentRole


@dataclass(frozen=True)
class ModelEndpoint:
    """A concrete LiteLLM model target (alias or openai/<name> + optional api_base)."""

    model: str
    api_base: str | None = None
    api_key: str | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "api_base": self.api_base,
            "label": self.label or self.model,
            "api_key_set": bool(self.api_key),
        }


ROLE_KEYS = ("planner", "searcher", "recommender", "router")

# LiteLLM treats the segment before "/" as a provider. Org-scoped served model
# ids (nvidia/..., defog/..., NousResearch/...) must be openai/<id> when calling
# an OpenAI-compatible api_base (Cloudera AI Inference / NIM).
_KNOWN_LITELLM_PREFIXES = (
    "openai/",
    "azure/",
    "hosted_vllm/",
    "nvidia_nim/",
    "huggingface/",
    "ollama/",
    "bedrock/",
    "vertex_ai/",
    "anthropic/",
)


def as_openai_compatible_model(model: str, *, via_proxy: bool = False) -> str:
    """Normalize a served model id for LiteLLM OpenAI-compatible calls."""
    model = (model or "").strip()
    if not model:
        return model
    if any(model.startswith(p) for p in _KNOWN_LITELLM_PREFIXES):
        return model
    # Proxy aliases are usually bare names (no slash); leave them alone.
    if via_proxy and "/" not in model:
        return model
    return f"openai/{model}"


def _default_profiles_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "config" / "model_profiles.yaml"


def load_profiles(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else Path(
        os.getenv("RCR_MODEL_PROFILES", str(_default_profiles_path()))
    )
    if not path.exists():
        return {"active_profile": "single-default", "profiles": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _endpoint_from_mapping(raw: dict[str, Any] | str, *, fallback: ModelEndpoint) -> ModelEndpoint:
    if isinstance(raw, str):
        return ModelEndpoint(
            model=raw, api_base=fallback.api_base, api_key=fallback.api_key, label=raw
        )
    model = raw.get("model") or fallback.model
    return ModelEndpoint(
        model=model,
        api_base=raw.get("api_base") or fallback.api_base,
        api_key=raw.get("api_key") or fallback.api_key,
        label=raw.get("label") or model,
    )


def _env_override(role: str, base: ModelEndpoint) -> ModelEndpoint:
    """LITELLM_MODEL_<ROLE> and optional CLOUDERA_*_<ROLE> / API_BASE_<ROLE>."""
    role_u = role.upper()
    model = os.getenv(f"LITELLM_MODEL_{role_u}") or os.getenv(f"CLOUDERA_AI_INFERENCE_MODEL_{role_u}")
    api_base = os.getenv(f"LITELLM_API_BASE_{role_u}") or os.getenv(
        f"CLOUDERA_AI_INFERENCE_API_BASE_{role_u}"
    )
    if not model and not api_base:
        return base
    return ModelEndpoint(
        model=model or base.model,
        api_base=(api_base or base.api_base),
        api_key=base.api_key,
        label=model or base.label,
    )


def _ensure_openai_prefix(model: str, *, via_proxy: bool, api_base: str | None = None) -> str:
    # LiteLLM proxy aliases stay bare (e.g. cloudera-planner).
    if via_proxy:
        return as_openai_compatible_model(model, via_proxy=True)
    # Direct OpenAI-compatible bases (CAIIS): always openai/<served-id>.
    if api_base:
        return as_openai_compatible_model(model, via_proxy=False)
    return as_openai_compatible_model(model, via_proxy=False)


class ModelRegistry:
    """Resolves which LLM each agent role (and optional router) should call."""

    def __init__(
        self,
        profile_name: str | None = None,
        profiles_doc: dict[str, Any] | None = None,
    ) -> None:
        self.doc = profiles_doc if profiles_doc is not None else load_profiles()
        self.profile_name = (
            profile_name
            or os.getenv("RCR_MODEL_PROFILE")
            or self.doc.get("active_profile")
            or "single-default"
        )
        self.settings = resolve_cloudera_settings()
        self._fallback = self._build_fallback()
        self._by_role = self._build_role_map()

    def _build_fallback(self) -> ModelEndpoint:
        if self.settings is not None:
            return ModelEndpoint(
                model=self.settings.litellm_chat_model,
                api_base=self.settings.api_base,
                api_key=self.settings.api_key,
                label=self.settings.chat_model,
            )
        return ModelEndpoint(
            model=os.getenv("LITELLM_MODEL", "gpt-4o-mini"),
            api_base=os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_API_BASE"),
            api_key=resolve_api_key(),
            label="default",
        )

    def _build_role_map(self) -> dict[str, ModelEndpoint]:
        via_proxy = bool(self.settings and self.settings.via_proxy)
        profiles = self.doc.get("profiles") or {}
        profile = profiles.get(self.profile_name) or {}
        roles_raw = profile.get("roles") or {}

        mapping: dict[str, ModelEndpoint] = {}
        for role in ROLE_KEYS:
            if role in roles_raw:
                ep = _endpoint_from_mapping(roles_raw[role], fallback=self._fallback)
            else:
                ep = self._fallback
            ep = _env_override(role, ep)
            mapping[role] = ModelEndpoint(
                model=_ensure_openai_prefix(
                    ep.model, via_proxy=via_proxy, api_base=ep.api_base
                ),
                api_base=ep.api_base,
                api_key=ep.api_key or self._fallback.api_key,
                label=ep.label or ep.model,
            )

        # User Settings discoveries override profile/env when available.
        use_user = os.getenv("RCR_USE_USER_ENDPOINTS", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        if use_user or self.profile_name == "user-discovered":
            try:
                from .capability_probe import assignments_to_model_endpoints
                from .endpoint_store import load_store

                discovered = assignments_to_model_endpoints(load_store())
                for role, ep in discovered.items():
                    if ep.model:
                        mapping[role] = ModelEndpoint(
                            model=_ensure_openai_prefix(
                                ep.model,
                                via_proxy=via_proxy,
                                api_base=ep.api_base or mapping[role].api_base,
                            ),
                            api_base=ep.api_base or mapping[role].api_base,
                            api_key=coerce_api_key(
                                ep.api_key,
                                fallback=mapping[role].api_key or self._fallback.api_key,
                            ),
                            label=ep.label or ep.model,
                        )
                if discovered:
                    self.profile_name = "user-discovered"
            except Exception:
                pass
        return mapping

    def for_role(self, role: str | AgentRole) -> ModelEndpoint:
        key = role.value if isinstance(role, AgentRole) else role
        return self._by_role.get(key, self._fallback)

    def role_model_map(self) -> dict[str, str]:
        return {role: self.for_role(role).model for role in ("planner", "searcher", "recommender")}

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "roles": {k: v.to_dict() for k, v in self._by_role.items()},
        }


def active_model_registry() -> ModelRegistry:
    return ModelRegistry()
