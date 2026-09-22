"""Token budget allocator B_i = β_base + β_role(R_i) (RCR-Router §2)."""

from __future__ import annotations

import os
from typing import Literal

from .models import AgentRole

BudgetPreset = Literal["paper", "savings", "demo"]

# Paper-scale bands (~512–4096 context budgets in experiments).
PAPER_BASE_BUDGET = 512
PAPER_ROLE_OFFSETS: dict[AgentRole, int] = {
    AgentRole.PLANNER: 1536,  # → 2048
    AgentRole.SEARCHER: 1024,  # → 1536
    AgentRole.RECOMMENDER: 1280,  # → 1792
}
PAPER_FULL_CONTEXT_CAP = 4096

# Demo / AMP compare: tighter B_i so RCR shows clear token savings vs full.
# Full-context cap stays paper-sized so the baseline remains fat.
SAVINGS_BASE_BUDGET = 256
SAVINGS_ROLE_OFFSETS: dict[AgentRole, int] = {
    AgentRole.PLANNER: 512,  # → 768
    AgentRole.SEARCHER: 256,  # → 512
    AgentRole.RECOMMENDER: 384,  # → 640
}
SAVINGS_FULL_CONTEXT_CAP = 4096

# Fast local / unit-test preset (word-era demo scale).
DEMO_BASE_BUDGET = 80
DEMO_ROLE_OFFSETS: dict[AgentRole, int] = {
    AgentRole.PLANNER: 100,  # → 180
    AgentRole.SEARCHER: 60,  # → 140
    AgentRole.RECOMMENDER: 80,  # → 160
}
DEMO_FULL_CONTEXT_CAP = 2000

# Back-compat aliases used by older imports/tests
DEFAULT_BASE_BUDGET = PAPER_BASE_BUDGET
DEFAULT_ROLE_OFFSETS = PAPER_ROLE_OFFSETS


def budget_preset() -> BudgetPreset:
    raw = os.getenv("RCR_BUDGET_PRESET", "savings").strip().lower()
    if raw in {"demo", "fast", "small"}:
        return "demo"
    if raw in {"paper", "full", "large"}:
        return "paper"
    return "savings"


class TokenBudgetAllocator:
    """B_i = β_base + β_role(R_i) from RCR-Router §2."""

    def __init__(
        self,
        base: int | None = None,
        role_offsets: dict[AgentRole, int] | None = None,
        *,
        preset: BudgetPreset | None = None,
        full_context_cap: int | None = None,
    ) -> None:
        preset = preset or budget_preset()
        if preset == "demo":
            default_base = DEMO_BASE_BUDGET
            default_offsets = DEMO_ROLE_OFFSETS
            default_full = DEMO_FULL_CONTEXT_CAP
        elif preset == "savings":
            default_base = SAVINGS_BASE_BUDGET
            default_offsets = SAVINGS_ROLE_OFFSETS
            default_full = SAVINGS_FULL_CONTEXT_CAP
        else:
            default_base = PAPER_BASE_BUDGET
            default_offsets = PAPER_ROLE_OFFSETS
            default_full = PAPER_FULL_CONTEXT_CAP

        env_base = os.getenv("RCR_BUDGET_BASE", "").strip()
        self.base = (
            base
            if base is not None
            else (int(env_base) if env_base.isdigit() else default_base)
        )
        self.role_offsets = role_offsets or dict(default_offsets)
        self.preset = preset
        self.full_context_cap = (
            full_context_cap
            if full_context_cap is not None
            else int(os.getenv("RCR_FULL_CONTEXT_CAP", str(default_full)))
        )

        # Optional multiplier (e.g. 0.75) applied after preset/base resolution.
        scale_raw = os.getenv("RCR_BUDGET_SCALE", "").strip()
        if scale_raw:
            try:
                scale = float(scale_raw)
            except ValueError:
                scale = 1.0
            if 0.1 <= scale <= 2.0 and scale != 1.0:
                self.base = max(32, int(round(self.base * scale)))
                self.role_offsets = {
                    role: max(0, int(round(off * scale)))
                    for role, off in self.role_offsets.items()
                }
                self.full_context_cap = max(
                    self.base, int(round(self.full_context_cap * scale))
                )

    @classmethod
    def from_env(cls) -> TokenBudgetAllocator:
        return cls(preset=budget_preset())

    def budget_for(self, role: AgentRole) -> int:
        return self.base + self.role_offsets.get(role, 0)

    def as_dict(self) -> dict:
        return {
            "preset": self.preset,
            "base": self.base,
            "full_context_cap": self.full_context_cap,
            "roles": {r.value: self.budget_for(r) for r in AgentRole},
        }
