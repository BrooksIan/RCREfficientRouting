"""Probe the knowledge store for grounding strength before agent runs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .knowledge_writeback import select_knowledge_for_context
from .models import MemoryItem
from .store import KnowledgeStore

KnowledgeStrength = Literal["strong", "medium", "weak"]

_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "are",
    "for",
    "with",
    "what",
    "which",
    "how",
    "do",
    "you",
    "name",
    "list",
}


@dataclass
class KnowledgeProbeResult:
    hits: list[MemoryItem] = field(default_factory=list)
    hit_count: int = 0
    top_score: float = 0.0
    lexical_coverage: float = 0.0
    strength: KnowledgeStrength = "weak"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("hits", None)
        d["hit_ids"] = [h.id for h in self.hits]
        return d


def _content_terms(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in _STOP
    }


def lexical_coverage(query: str, texts: list[str]) -> float:
    q = _content_terms(query)
    if not q:
        return 0.0
    pooled: set[str] = set()
    for t in texts:
        pooled |= _content_terms(t)
    return len(q & pooled) / len(q)


def classify_strength(
    *,
    hit_count: int,
    coverage: float,
) -> KnowledgeStrength:
    if hit_count >= 2 and coverage >= 0.25:
        return "strong"
    if hit_count == 0 or coverage < 0.1:
        return "weak"
    return "medium"


def probe_knowledge(
    knowledge: KnowledgeStore,
    query: str,
    *,
    embedding: list[float] | None = None,
    query_task: str | None = None,
    k: int = 8,
    max_items: int = 3,
) -> KnowledgeProbeResult:
    """Search + filter knowledge; return strength signal and reusable hits."""
    raw = knowledge.search(
        query,
        k=k,
        embedding=embedding,
        task=query_task,
        soft_task=True,
    )
    hits = select_knowledge_for_context(raw, query=query, max_items=max_items)
    coverage = lexical_coverage(query, [h.text or "" for h in hits])
    top = max((float(h.score or 0.0) for h in hits), default=0.0)
    strength = classify_strength(hit_count=len(hits), coverage=coverage)
    return KnowledgeProbeResult(
        hits=hits,
        hit_count=len(hits),
        top_score=round(top, 4),
        lexical_coverage=round(coverage, 4),
        strength=strength,
    )
