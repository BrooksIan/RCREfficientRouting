"""Persist user-supplied LLM endpoints and discovered role assignments."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


# Operator-declared purpose for each model (influences capability priors).
INTENDED_TASKS: tuple[str, ...] = (
    "generic",
    "coding",
    "reasoning",
    "extraction",
    "synthesis",
    "routing",
)

INTENDED_TASK_LABELS: dict[str, str] = {
    "generic": "Generic — all-purpose chat",
    "coding": "Coding — code / tools / structured edits",
    "reasoning": "Reasoning — planning & multi-step logic",
    "extraction": "Extraction — facts, search, grounding",
    "synthesis": "Synthesis — answers & recommendations",
    "routing": "Routing — classify / gate / JSON control",
}


def default_store_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    override = os.getenv("RCR_USER_ENDPOINTS_PATH")
    if override:
        return Path(override)
    return root / ".rcr" / "user_endpoints.json"


@dataclass
class UserEndpoint:
    slot: int  # 1..3
    label: str = ""
    model_id: str = ""
    api_base: str = ""
    api_key: str = ""
    intended_task: str = "generic"
    enabled: bool = True

    def normalized_task(self) -> str:
        task = (self.intended_task or "generic").strip().lower()
        return task if task in INTENDED_TASKS else "generic"

    def is_configured(self) -> bool:
        return bool(self.model_id.strip() and self.api_base.strip())

    def to_public_dict(self) -> dict[str, Any]:
        """Safe for UI display (masks api key)."""
        return {
            "slot": self.slot,
            "label": self.label,
            "model_id": self.model_id,
            "api_base": self.api_base,
            "intended_task": self.normalized_task(),
            "api_key_set": bool(self.api_key),
            "api_key_preview": ("••••" + self.api_key[-4:]) if len(self.api_key) >= 4 else "",
            "enabled": self.enabled,
        }


@dataclass
class CapabilityScores:
    planner: float = 0.0
    searcher: float = 0.0
    recommender: float = 0.0
    router: float = 0.0
    latency_sec: float = 0.0
    notes: dict[str, str] = field(default_factory=dict)
    samples: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EndpointStoreData:
    endpoints: list[UserEndpoint] = field(default_factory=list)
    capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)  # model_id -> scores
    role_assignments: dict[str, dict[str, Any]] = field(default_factory=dict)  # role -> endpoint info
    last_probed_at: str = ""

    def configured_endpoints(self) -> list[UserEndpoint]:
        return [e for e in self.endpoints if e.enabled and e.is_configured()]


def blank_endpoints() -> list[UserEndpoint]:
    return [UserEndpoint(slot=i, label=f"Model {i}") for i in range(1, 4)]


def _endpoint_from_dict(raw: dict[str, Any]) -> UserEndpoint:
    allowed = {f.name for f in fields(UserEndpoint)}
    return UserEndpoint(**{k: v for k, v in raw.items() if k in allowed})


def load_store(path: Path | None = None) -> EndpointStoreData:
    path = path or default_store_path()
    if not path.exists():
        return EndpointStoreData(endpoints=blank_endpoints())
    raw = json.loads(path.read_text(encoding="utf-8"))
    endpoints = [_endpoint_from_dict(e) for e in raw.get("endpoints") or blank_endpoints()]
    # ensure 3 slots
    by_slot = {e.slot: e for e in endpoints}
    endpoints = [by_slot.get(i) or UserEndpoint(slot=i, label=f"Model {i}") for i in range(1, 4)]
    return EndpointStoreData(
        endpoints=endpoints,
        capabilities=raw.get("capabilities") or {},
        role_assignments=raw.get("role_assignments") or {},
        last_probed_at=raw.get("last_probed_at") or "",
    )


def save_store(data: EndpointStoreData, path: Path | None = None) -> Path:
    path = path or default_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "endpoints": [asdict(e) for e in data.endpoints],
        "capabilities": data.capabilities,
        "role_assignments": data.role_assignments,
        "last_probed_at": data.last_probed_at,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
