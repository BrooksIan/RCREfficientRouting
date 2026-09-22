"""Optional LLM that chooses which agents to run this round."""

from __future__ import annotations

import json
import re

from .llm import LLMClient
from .model_routing import ModelEndpoint, ModelRegistry
from .models import AgentRole


DEFAULT_ORDER = [AgentRole.PLANNER, AgentRole.SEARCHER, AgentRole.RECOMMENDER]


class AgentActivationRouter:
    """Uses a dedicated router LLM (or heuristic) to pick active agents."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ModelRegistry | None = None,
        enabled: bool | None = None,
    ) -> None:
        import os

        self.llm = llm
        self.registry = registry or ModelRegistry()
        if enabled is None:
            enabled = os.getenv("RCR_LLM_AGENT_ROUTER", "false").lower() in {
                "1",
                "true",
                "yes",
            }
        self.enabled = enabled

    def select(
        self,
        query: str,
        *,
        round_idx: int,
        prior_roles: list[str] | None = None,
    ) -> list[AgentRole]:
        if not self.enabled:
            return list(DEFAULT_ORDER)

        endpoint = self.registry.for_role("router")
        prior = ", ".join(prior_roles or []) or "(none)"
        prompt = (
            "You route work in a multi-agent system with roles: planner, searcher, recommender.\n"
            "Return ONLY a JSON list of roles to run this round, e.g. "
            '["planner","searcher","recommender"].\n'
            "Include planner early if the task is underspecified; include searcher if evidence "
            "is needed; include recommender when enough context exists to answer.\n"
            f"Round: {round_idx}\nPrior roles this session: {prior}\n"
            f"User query: {query}\n"
        )
        try:
            resp = self.llm.complete(
                [
                    {
                        "role": "system",
                        "content": "You are a routing controller. Output JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=endpoint.model,
                api_base=endpoint.api_base,
                api_key=endpoint.api_key,
            )
            roles = self._parse_roles(resp.text)
            if roles:
                return roles
        except Exception:
            pass
        return list(DEFAULT_ORDER)

    def _parse_roles(self, text: str) -> list[AgentRole]:
        text = (text or "").strip()
        match = re.search(r"\[[^\]]+\]", text)
        if not match:
            return []
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        out: list[AgentRole] = []
        for item in raw:
            name = str(item).strip().lower()
            try:
                role = AgentRole(name)
            except ValueError:
                continue
            if role not in out:
                out.append(role)
        return out
