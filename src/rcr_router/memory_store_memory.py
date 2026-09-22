from __future__ import annotations

import math
from typing import Optional

from .models import MemoryItem
from .store import KnowledgeStore, MemoryStore
from .task_classify import task_match_score


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _lexical(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def _apply_task_boost(score: float, item_task: str | None, query_task: str | None, *, soft: bool) -> float:
    if not query_task or query_task == "generic":
        return score
    match = task_match_score(item_task, query_task)
    if soft:
        return score * (0.55 + 0.7 * match)
    # Hard preference: keep match / generic / untagged; bury mismatches
    if match >= 0.55:
        return score * (0.7 + 0.5 * match)
    return score * 0.15


class InMemoryMemoryStore(MemoryStore):
    """Test double for OpenSearch working memory."""

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def upsert(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    def bulk_upsert(self, items: list[MemoryItem]) -> None:
        for item in items:
            self.upsert(item)

    def list_session(self, session_id: str) -> list[MemoryItem]:
        return [i for i in self._items.values() if i.session_id == session_id]

    def delete(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    def search(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        role_tag: Optional[str] = None,
        stage_tag: Optional[str] = None,
        task: Optional[str] = None,
        k: int = 20,
        embedding: Optional[list[float]] = None,
    ) -> list[MemoryItem]:
        candidates = list(self._items.values())
        if session_id is not None:
            candidates = [i for i in candidates if i.session_id == session_id]
        if role_tag is not None:
            candidates = [i for i in candidates if i.role_tag == role_tag]
        if stage_tag is not None:
            candidates = [i for i in candidates if i.stage_tag == stage_tag]

        scored: list[MemoryItem] = []
        for item in candidates:
            lex = _lexical(query, item.text)
            sem = _cosine(embedding or [], item.embedding or []) if embedding else 0.0
            hit = MemoryItem.from_dict(item.to_dict())
            hit.score = _apply_task_boost(0.5 * lex + 0.5 * sem, item.task, task, soft=True)
            scored.append(hit)
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]


class InMemoryKnowledgeStore(KnowledgeStore):
    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def upsert(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    def bulk_upsert(self, items: list[MemoryItem]) -> None:
        for item in items:
            self.upsert(item)

    def delete(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        embedding: Optional[list[float]] = None,
        task: Optional[str] = None,
        soft_task: bool = True,
    ) -> list[MemoryItem]:
        scored: list[MemoryItem] = []
        for item in self._items.values():
            if (
                not soft_task
                and task
                and task != "generic"
                and item.task
                and item.task not in {task, "generic"}
            ):
                continue
            lex = _lexical(query, item.text)
            sem = _cosine(embedding or [], item.embedding or []) if embedding else 0.0
            hit = MemoryItem.from_dict(item.to_dict())
            hit.score = _apply_task_boost(0.5 * lex + 0.5 * sem, item.task, task, soft=soft_task)
            scored.append(hit)
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]
