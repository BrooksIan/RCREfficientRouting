"""Per-run task + knowledge → model plan (opt-in via RCR_TASK_MODEL_ROUTING)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .cloudera import coerce_api_key
from .endpoint_store import EndpointStoreData, UserEndpoint, load_store
from .knowledge_probe import KnowledgeProbeResult, KnowledgeStrength
from .model_routing import ModelEndpoint, ModelRegistry, as_openai_compatible_model
from .models import AgentRole

ROLE_KEYS = ("planner", "searcher", "recommender")


def task_model_routing_enabled(override: bool | None = None) -> bool:
    """User/env opt-in. Default off. Explicit override wins for a single run."""
    if override is not None:
        return bool(override)
    return os.getenv("RCR_TASK_MODEL_ROUTING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "config" / "task_model_policy.yaml"


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else Path(
        os.getenv("RCR_TASK_MODEL_POLICY", str(_default_policy_path()))
    )
    if not path.exists():
        return {"version": 1, "tasks": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"version": 1, "tasks": {}}


@dataclass
class RoleModelChoice:
    role: str
    endpoint: ModelEndpoint
    reason: str
    intended_task: str = ""
    capability_score: float = 0.0
    latency_sec: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.endpoint.model,
            "api_base": self.endpoint.api_base,
            "label": self.endpoint.label,
            "reason": self.reason,
            "intended_task": self.intended_task,
            "capability_score": self.capability_score,
            "latency_sec": self.latency_sec,
            "api_key_set": bool(self.endpoint.api_key),
        }


@dataclass
class RunModelPlan:
    enabled: bool
    task: str
    knowledge_strength: KnowledgeStrength | str
    choices: dict[str, RoleModelChoice] = field(default_factory=dict)
    note: str = ""

    def for_role(self, role: str | AgentRole) -> ModelEndpoint | None:
        key = role.value if isinstance(role, AgentRole) else role
        choice = self.choices.get(key)
        return choice.endpoint if choice else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "task": self.task,
            "knowledge_strength": self.knowledge_strength,
            "note": self.note,
            "roles": {k: v.as_dict() for k, v in self.choices.items()},
        }


def _endpoint_from_user(ep: UserEndpoint) -> ModelEndpoint:
    return ModelEndpoint(
        model=as_openai_compatible_model(ep.model_id),
        api_base=ep.api_base or None,
        api_key=coerce_api_key(ep.api_key) or ep.api_key or None,
        label=ep.label or ep.model_id,
    )


def _capability(store: EndpointStoreData, ep: UserEndpoint, role: str) -> tuple[float, float]:
    cap = store.capabilities.get(ep.model_id) or store.capabilities.get(str(ep.slot)) or {}
    score = float(cap.get(role, 0.0) or 0.0)
    latency = float(cap.get("latency_sec", 0.0) or 0.0)
    return score, latency


def _pick_for_role(
    *,
    role: str,
    prefer_task: str | None,
    strength: KnowledgeStrength,
    escalate: bool,
    store: EndpointStoreData,
    fallback: ModelEndpoint,
) -> RoleModelChoice:
    configured = store.configured_endpoints()
    if not configured:
        return RoleModelChoice(
            role=role,
            endpoint=fallback,
            reason="fallback_registry_no_user_endpoints",
        )

    scored: list[tuple[float, float, float, UserEndpoint, str]] = []
    # tuple: sort_key parts → (match_bonus, capability, -latency), ep, reason_stem
    for ep in configured:
        score, latency = _capability(store, ep, role)
        intended = ep.normalized_task()
        match = 1.0 if prefer_task and intended == prefer_task else 0.0
        if prefer_task and intended == "generic":
            match = max(match, 0.35)
        scored.append((match, score, latency, ep, intended))

    if escalate or strength == "weak":
        # Best capability for this role; ignore intended_task soft preference.
        scored.sort(key=lambda row: (row[1], -row[2]), reverse=True)
        best = scored[0]
        ep = best[3]
        return RoleModelChoice(
            role=role,
            endpoint=_endpoint_from_user(ep),
            reason=f"escalate_weak_knowledge capability={best[1]:.2f}",
            intended_task=best[4],
            capability_score=best[1],
            latency_sec=best[2],
        )

    if strength == "strong":
        # Prefer matches, then lower latency, then capability.
        matched = [r for r in scored if r[0] >= 1.0] or scored
        matched.sort(key=lambda row: (row[0], -row[2], row[1]), reverse=True)
        # Among top match tier, pick lowest latency (sort already: match desc, -lat desc means high -lat first = low lat)
        # Wait: reverse=True on (-latency) means more negative first = higher latency wins. Fix:
        matched.sort(key=lambda row: (-row[0], row[2], -row[1]))
        best = matched[0]
        ep = best[3]
        return RoleModelChoice(
            role=role,
            endpoint=_endpoint_from_user(ep),
            reason=f"strong_knowledge cheap_match intended={best[4]} latency={best[2]:.2f}",
            intended_task=best[4],
            capability_score=best[1],
            latency_sec=best[2],
        )

    # medium: intended_task match, then capability
    scored.sort(key=lambda row: (row[0], row[1], -row[2]), reverse=True)
    best = scored[0]
    ep = best[3]
    reason = (
        f"task_match intended={best[4]}"
        if best[0] >= 1.0
        else f"best_available intended={best[4]} cap={best[1]:.2f}"
    )
    return RoleModelChoice(
        role=role,
        endpoint=_endpoint_from_user(ep),
        reason=reason,
        intended_task=best[4],
        capability_score=best[1],
        latency_sec=best[2],
    )


def build_run_model_plan(
    *,
    query_task: str,
    probe: KnowledgeProbeResult | None,
    registry: ModelRegistry,
    store: EndpointStoreData | None = None,
    policy_doc: dict[str, Any] | None = None,
    enabled: bool | None = None,
) -> RunModelPlan:
    """Build per-role endpoints for this run. No-op when routing disabled."""
    if not task_model_routing_enabled(enabled):
        return RunModelPlan(
            enabled=False,
            task=query_task or "generic",
            knowledge_strength=(probe.strength if probe else "weak"),
            note="routing_disabled",
        )

    store = store if store is not None else load_store()
    configured = store.configured_endpoints()
    if len(configured) < 2:
        return RunModelPlan(
            enabled=False,
            task=query_task or "generic",
            knowledge_strength=(probe.strength if probe else "weak"),
            note="need_at_least_2_endpoints",
        )

    doc = policy_doc if policy_doc is not None else load_policy()
    task = (query_task or "generic").strip().lower()
    task_cfg = (doc.get("tasks") or {}).get(task) or (doc.get("tasks") or {}).get("generic") or {}
    prefer_map: dict[str, str] = dict(task_cfg.get("prefer") or {})
    escalate_roles = set(task_cfg.get("escalate_roles") or ROLE_KEYS)
    strength: KnowledgeStrength = probe.strength if probe else "medium"

    choices: dict[str, RoleModelChoice] = {}
    for role in ROLE_KEYS:
        fallback = registry.for_role(role)
        prefer = prefer_map.get(role)
        escalate = strength == "weak" and role in escalate_roles
        choices[role] = _pick_for_role(
            role=role,
            prefer_task=prefer,
            strength=strength,
            escalate=escalate,
            store=store,
            fallback=fallback,
        )

    return RunModelPlan(
        enabled=True,
        task=task,
        knowledge_strength=strength,
        choices=choices,
        note="ok",
    )
