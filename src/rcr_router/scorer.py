from __future__ import annotations

import math
import time
from typing import Iterable

from .models import ROLE_KEYWORDS, AgentRole, DocType, MemoryItem, TaskStage
from .task_classify import task_match_score


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, (dot / (na * nb) + 1.0) / 2.0))  # map [-1,1] -> [0,1]


class ImportanceScorer:
    """α(m; R_i, S_t, task): retrieval + embedding + role + stage + recency + task."""

    def __init__(
        self,
        w_retrieval: float = 0.26,
        w_embedding: float = 0.22,
        w_role: float = 0.18,
        w_stage: float = 0.12,
        w_recency: float = 0.08,
        w_task: float = 0.14,
        now: float | None = None,
    ) -> None:
        self.w_retrieval = w_retrieval
        self.w_embedding = w_embedding
        self.w_role = w_role
        self.w_stage = w_stage
        self.w_recency = w_recency
        self.w_task = w_task
        self.now = now

    def score(
        self,
        item: MemoryItem,
        role: AgentRole,
        stage: TaskStage,
        retrieval_score: float = 0.0,
        query_embedding: list[float] | None = None,
        query_task: str | None = None,
    ) -> float:
        role_score = self._role_relevance(item, role)
        stage_score = 1.0 if item.stage_tag == stage.value else (
            0.5 if item.stage_tag is None else 0.2
        )
        recency = self._recency(item.timestamp)
        ret = 1.0 - math.exp(-max(0.0, retrieval_score))
        emb = _cosine(query_embedding, item.embedding)
        task_score = task_match_score(item.task, query_task)
        return (
            self.w_retrieval * ret
            + self.w_embedding * emb
            + self.w_role * role_score
            + self.w_stage * stage_score
            + self.w_recency * recency
            + self.w_task * task_score
        )

    def score_many(
        self,
        items: Iterable[MemoryItem],
        role: AgentRole,
        stage: TaskStage,
        query_embedding: list[float] | None = None,
        query_task: str | None = None,
    ) -> list[MemoryItem]:
        scored: list[MemoryItem] = []
        for item in items:
            clone = MemoryItem.from_dict(item.to_dict())
            clone.score = self.score(
                clone,
                role,
                stage,
                retrieval_score=item.score,
                query_embedding=query_embedding,
                query_task=query_task,
            )
            scored.append(clone)
        return scored

    def _role_relevance(self, item: MemoryItem, role: AgentRole) -> float:
        text = item.text or ""
        lower = text.lower()
        # Recommender must see evidence/facts, not only texts containing "answer".
        if role == AgentRole.RECOMMENDER:
            if lower.startswith("user_query:"):
                return 1.0
            if item.doc_type in {
                DocType.FACT.value,
                DocType.KNOWLEDGE.value,
                DocType.PLAN.value,
            }:
                return 0.95
            if item.source_agent in {"searcher", "knowledge", "planner"}:
                return 0.9
            if item.role_tag in {
                AgentRole.SEARCHER.value,
                AgentRole.PLANNER.value,
                "user",
            }:
                return 0.85
        keywords = ROLE_KEYWORDS.get(role, [])
        if not keywords:
            return 0.0
        hits = sum(1 for kw in keywords if kw in lower)
        return min(1.0, hits / max(1, len(keywords) * 0.5))

    def _recency(self, timestamp: float) -> float:
        now = self.now if self.now is not None else time.time()
        age_hours = max(0.0, (now - timestamp) / 3600.0)
        return math.exp(-age_hours / 24.0)
