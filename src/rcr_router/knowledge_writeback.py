"""Write agent findings back into the durable knowledge corpus (deduped)."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable

from .models import AgentRole, DocType, MemoryItem, TaskStage
from .store import KnowledgeStore

# Prefer recommender answers; searcher facts only if no answer exists yet.
WRITEBACK_ROLES = {AgentRole.SEARCHER, AgentRole.RECOMMENDER}

_LOW_VALUE = (
    "does not contain",
    "no concrete facts",
    "no information",
    "no relevant evidence",
    "no relevant",
    "not present in",
    "cannot answer from",
    "i don't know",
    "i do not know",
    "[agent error",
    "task_record:",
    "supplied context does not",
    "provided context does not",
    "provided material that answer",
    "appears in the provided context",
)

# Planner/process chrome and instruction leakage — never durable knowledge.
_META_LOW_VALUE = (
    "thus output",
    "output something like",
    "output only the plan",
    "we should only output",
    "no extra text",
    "no chain-of-thought",
    "no chain of thought",
    "present the results",
    "be concise",
    "follow instructions",
    "follow the instructions",
    "do not include",
    "remember:",
    "search intents",
    "decompose the user query",
    "use only the provided context",
    "routed context",
    "working notes",
    "self-checks",
    "citation-only",
    "kind: plan_step",
    "evidence_ids: []",
)


def _strip_writeback_prefix(text: str) -> str:
    return re.sub(
        r"^(?:answer|fact)\s*\(\s*query:\s*[^)]*\)\s*:\s*",
        "",
        (text or "").strip(),
        flags=re.I,
    ).strip()


def _has_factual_substance(text: str) -> bool:
    """True when text looks like durable facts (lists, quantities, concrete claims)."""
    t = (text or "").strip()
    if re.search(r"\b\d+\s*,\s*\d+\s*,\s*\d+", t):
        return True
    if re.search(r"\b\d+(?:\.\d+)?%\b", t):
        return True
    # Named claim-ish sentences with a verb and enough content words
    words = [w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "plan",
        "step",
        "output",
        "only",
        "query",
        "context",
        "should",
        "would",
        "could",
    }
    content = [w for w in words if w not in stop]
    if len(content) >= 8 and re.search(
        r"\b(is|are|was|were|has|have|had|includes?|contains?|equals?|drove|caused)\b",
        t,
        re.I,
    ):
        return True
    return False


def is_low_value(text: str) -> bool:
    """Skip refusals, meta/instruction chrome, and empty evidence — keep durable facts."""
    raw = (text or "").strip()
    if not raw:
        return True
    t = _normalize(raw)
    body = _normalize(_strip_writeback_prefix(raw))
    if len(body) < 24:
        return True

    # Citation-only / markup-only
    if re.fullmatch(r"(?:\s*(?:\[\s*\d+\s*\]|【[^】]+】)\s*)+", raw):
        return True

    # Meta / process leakage (unless the same blob also has strong factual substance)
    if any(p in body for p in _META_LOW_VALUE) and not _has_factual_substance(body):
        return True

    has_refusal = any(p in body for p in _LOW_VALUE)
    has_substance = _has_factual_substance(body) or (
        len(body.split()) >= 25
        and not body.startswith(("no ", "the supplied", "the provided"))
    )
    if has_refusal and not has_substance:
        return True
    if has_refusal and re.match(
        r"^(no |fact \(query:.*\): no |the supplied|the provided context)",
        t,
    ):
        return True

    # Pure plan outlines with no facts
    if re.match(r"^(?:plan|steps?)\s*[:\-]", body) and not _has_factual_substance(body):
        return True

    return False


_STRIP_PATTERNS = (
    re.compile(r"^(fact|answer)\s*\(query:\s*[^)]+\):\s*", re.I),
    re.compile(r"\[agent error[^\]]*\]", re.I),
    re.compile(r"from\s+\*\*?\[?\d+\]?\*\*?:?\s*", re.I),
    re.compile(r"from\s+\[\d+\]:?\s*", re.I),
    re.compile(r"\*\*"),
)


@dataclass
class WritebackResult:
    written: bool
    item_id: str = ""
    reason: str = ""


@dataclass
class CompactionResult:
    deleted: int = 0
    kept_id: str = ""
    deleted_ids: list[str] | None = None

    def as_dict(self) -> dict:
        return {
            "deleted": self.deleted,
            "kept_id": self.kept_id,
            "deleted_ids": list(self.deleted_ids or []),
        }


@dataclass
class CorpusCompactionResult:
    deleted: int = 0
    queries_compacted: int = 0
    remaining_writebacks: int = 0
    details: list[dict] | None = None

    def as_dict(self) -> dict:
        return {
            "deleted": self.deleted,
            "queries_compacted": self.queries_compacted,
            "remaining_writebacks": self.remaining_writebacks,
            "details": list(self.details or []),
        }


def writeback_enabled() -> bool:
    return os.getenv("RCR_KNOWLEDGE_WRITEBACK", "true").lower() in {"1", "true", "yes"}


def compact_mode() -> str:
    """Post-prompt compaction mode: off | sync | background (default)."""
    raw = os.getenv("RCR_COMPACT_AFTER_PROMPT", "background").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "sync"}:
        return "sync"
    # background | async | bg | "" 
    return "background"


def compact_after_prompt_enabled() -> bool:
    """True when sync compaction should run inside Orchestrator.run()."""
    return compact_mode() == "sync"


def route_near_dedupe_enabled() -> bool:
    """Near-dupe in the RCR hot path. Default off — budget fill is enough; near-dupe is O(n²)."""
    return os.getenv("RCR_ROUTE_NEAR_DEDUPE", "false").lower() in {"1", "true", "yes"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def content_id(text: str, *, prefix: str = "learned") -> str:
    digest = hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def query_slot_id(query: str) -> str:
    """One durable knowledge slot per normalized user query."""
    return content_id(_normalize(query), prefix="learned-q")


def canonicalize_fact_text(text: str) -> str:
    """Remove wrappers / markdown so paraphrases collide."""
    t = text or ""
    for pat in _STRIP_PATTERNS:
        t = pat.sub("", t)
    t = _normalize(t)
    # Soften list punctuation so "2, 3, 5" == "2,3,5"
    t = re.sub(r"\s*,\s*", ",", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .;:")


def _lexical_overlap(a: str, b: str) -> float:
    wa = set(canonicalize_fact_text(a).replace(",", " ").split())
    wb = set(canonicalize_fact_text(b).replace(",", " ").split())
    # drop ultra-common glue words
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "that", "this", "for", "with"}
    wa -= stop
    wb -= stop
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


def _jaccard(a: str, b: str) -> float:
    wa = set(canonicalize_fact_text(a).replace(",", " ").split())
    wb = set(canonicalize_fact_text(b).replace(",", " ").split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def near_duplicate(
    a: str,
    b: str,
    *,
    emb_a: list[float] | None = None,
    emb_b: list[float] | None = None,
    lexical_thresh: float = 0.55,
    jaccard_thresh: float = 0.42,
    cosine_thresh: float = 0.92,
) -> bool:
    if canonicalize_fact_text(a) == canonicalize_fact_text(b):
        return True
    if _lexical_overlap(a, b) >= lexical_thresh:
        return True
    if _jaccard(a, b) >= jaccard_thresh:
        return True
    if emb_a and emb_b and _cosine(emb_a, emb_b) >= cosine_thresh:
        return True
    return False


def _quality_score(text: str, role: AgentRole) -> float:
    """Higher = preferred canonical writeback (prefer answers, prefer concise)."""
    t = canonicalize_fact_text(text)
    words = max(1, len(t.split()))
    # Prefer recommender; penalize very long dumps; bonus for structured lists
    role_bonus = 2.0 if role == AgentRole.RECOMMENDER else 0.0
    length_pen = max(0.0, (words - 80) / 80.0)
    list_bonus = 0.5 if re.search(r"\d+,\d+,\d+", t) else 0.0
    return role_bonus + list_bonus - length_pen


class KnowledgeWriteback:
    """Persist novel agent outputs — at most one canonical doc per query."""

    def __init__(
        self,
        knowledge: KnowledgeStore,
        *,
        novelty_overlap: float = 0.55,
        enabled: bool | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.novelty_overlap = novelty_overlap
        self.enabled = writeback_enabled() if enabled is None else enabled
        self.writes: list[dict] = []

    def maybe_write(
        self,
        *,
        text: str,
        role: AgentRole,
        query: str,
        session_id: str,
        round_idx: int,
        embedding: list[float] | None,
        task: str | None = None,
    ) -> WritebackResult:
        if not self.enabled:
            return WritebackResult(False, reason="disabled")
        if role not in WRITEBACK_ROLES:
            return WritebackResult(False, reason="role_skipped")
        cleaned = (text or "").strip()
        if is_low_value(cleaned):
            return WritebackResult(False, reason="low_value")
        # Never persist citation-only answers like "[2] [3]"
        if re.fullmatch(r"(?:\s*\[\s*\d+\s*\]\s*)+", cleaned):
            return WritebackResult(False, reason="citation_only")

        slot_id = query_slot_id(query)
        existing = self._get_by_id(slot_id)
        new_score = _quality_score(cleaned, role)

        if existing is not None:
            # Same query slot: only replace if clearly better
            old_role = AgentRole.RECOMMENDER if "ANSWER" in (existing.text or "")[:12] else AgentRole.SEARCHER
            old_score = _quality_score(existing.text or "", old_role)
            if near_duplicate(cleaned, existing.text or "", emb_a=embedding, emb_b=existing.embedding):
                if new_score <= old_score + 0.15:
                    return WritebackResult(False, item_id=slot_id, reason="duplicate_slot")
            elif new_score <= old_score:
                return WritebackResult(False, item_id=slot_id, reason="weaker_than_slot")

        # Also skip if a near-duplicate exists under another id (legacy spam)
        if self._find_near_duplicate(cleaned, embedding=embedding, task=task, query=query):
            return WritebackResult(False, item_id=slot_id, reason="duplicate")

        prefix = "ANSWER" if role == AgentRole.RECOMMENDER else "FACT"
        # Keep body compact — no multi-paragraph dumps
        compact = cleaned
        if len(compact.split()) > 120:
            compact = " ".join(compact.split()[:120]) + "…"
        body = f"{prefix} (query: {query.strip()[:120]}): {compact}"

        item = MemoryItem(
            id=slot_id,
            text=body,
            role_tag=AgentRole.SEARCHER.value,
            stage_tag=TaskStage.SEARCHING.value,
            source_agent=role.value,
            session_id=session_id,
            round=round_idx,
            doc_type=DocType.KNOWLEDGE.value,
            embedding=embedding,
            task=task,
            metadata={
                "task": task,
                "kind": "agent_writeback",
                "source_role": role.value,
                "query": query[:300],
                "canonical": True,
            },
        )
        self.knowledge.upsert(item)
        self.writes.append(
            {
                "id": slot_id,
                "role": role.value,
                "task": task,
                "chars": len(compact),
                "replaced": existing is not None,
            }
        )
        return WritebackResult(True, item_id=slot_id, reason="written")

    def compact_after_prompt(
        self,
        query: str,
        *,
        embedding: list[float] | None = None,
        task: str | None = None,
        force: bool = False,
    ) -> CompactionResult:
        """After all agents finish for a prompt: keep one canonical writeback, drop dups.

        Seeded corpus docs are never deleted. Only agent_writeback / legacy learned-*
        / TASK_RECORD docs related to this query are compacted.

        Disabled unless mode is sync (`RCR_COMPACT_AFTER_PROMPT=sync|true`).
        Prefer background scheduling via ``schedule_compact_after_prompt`` so the
        UI can return answers first. Pass ``force=True`` for manual sweeps.
        """
        if not force and not compact_after_prompt_enabled():
            return CompactionResult()
        if not query.strip():
            return CompactionResult()

        qn = _normalize(query)
        slot_id = query_slot_id(query)
        hits = self.knowledge.search(
            query,
            k=25,
            embedding=embedding,
            task=task,
            soft_task=True,
        )
        # Include in-memory writebacks that search may miss
        if hasattr(self.knowledge, "_items"):
            seen = {h.id for h in hits}
            for item in list(getattr(self.knowledge, "_items").values()):
                if item.id in seen:
                    continue
                kind = (item.metadata or {}).get("kind")
                if item.id == slot_id or kind in {"agent_writeback", "task_record"}:
                    hits.append(item)
                    seen.add(item.id)

        candidates: list[MemoryItem] = []
        for hit in hits:
            meta = hit.metadata or {}
            kind = meta.get("kind")
            text = hit.text or ""
            if kind == "task_record" or text.upper().startswith("TASK_RECORD:"):
                if qn[:32] in _normalize(text) or _normalize(str(meta.get("query") or "")) == qn:
                    candidates.append(hit)
                continue
            is_wb = kind == "agent_writeback" or str(hit.id).startswith("learned")
            if not is_wb:
                continue
            meta_q = _normalize(str(meta.get("query") or ""))
            if meta_q == qn or hit.id == slot_id:
                candidates.append(hit)
            elif meta_q and near_duplicate(meta_q, qn, lexical_thresh=0.85):
                candidates.append(hit)

        if not candidates:
            # Still ensure slot is the only copy if present alone
            kept = self._get_by_id(slot_id)
            return CompactionResult(deleted=0, kept_id=kept.id if kept else "", deleted_ids=[])

        def rank(item: MemoryItem) -> tuple:
            meta = item.metadata or {}
            text = item.text or ""
            role = AgentRole.RECOMMENDER if text.startswith("ANSWER") else AgentRole.SEARCHER
            return (
                0 if meta.get("canonical") else 1,
                0 if text.startswith("ANSWER") else 1,
                -_quality_score(text, role),
                len(text),
            )

        # Drop task records immediately; rank writebacks
        writebacks = [
            c
            for c in candidates
            if (c.metadata or {}).get("kind") != "task_record"
            and not (c.text or "").upper().startswith("TASK_RECORD:")
        ]
        tasks = [c for c in candidates if c not in writebacks]

        deleted_ids: list[str] = []
        for t in tasks:
            self._delete(t.id)
            deleted_ids.append(t.id)

        if not writebacks:
            return CompactionResult(deleted=len(deleted_ids), kept_id="", deleted_ids=deleted_ids)

        writebacks.sort(key=rank)
        winner = writebacks[0]
        # Promote winner to stable slot id
        winner_copy = MemoryItem.from_dict(winner.to_dict())
        winner_copy.id = slot_id
        meta = dict(winner_copy.metadata or {})
        meta["canonical"] = True
        meta["query"] = query[:300]
        meta["kind"] = "agent_writeback"
        winner_copy.metadata = meta
        self.knowledge.upsert(winner_copy)

        for item in writebacks:
            if item.id == slot_id:
                continue
            self._delete(item.id)
            deleted_ids.append(item.id)

        return CompactionResult(
            deleted=len(deleted_ids),
            kept_id=slot_id,
            deleted_ids=deleted_ids,
        )

    def compact_corpus(self) -> CorpusCompactionResult:
        """Force-compact the whole knowledge corpus (Settings / manual sweep).

        Keeps seeded docs. Collapses agent writebacks to one `learned-q-*` per query
        and removes leftover TASK_RECORD docs. Always runs (ignores env gate).
        """
        items = _list_knowledge_items(self.knowledge)
        queries: set[str] = set()
        for item in items:
            meta = item.metadata or {}
            q = str(meta.get("query") or "").strip()
            if q:
                queries.add(q)
            text = item.text or ""
            if text.upper().startswith("TASK_RECORD:") and "Query:" in text:
                queries.add(text.split("Query:", 1)[-1].strip())

        details: list[dict] = []
        total_deleted = 0
        for q in sorted(queries):
            result = self.compact_after_prompt(q, force=True)
            total_deleted += result.deleted
            if result.deleted or result.kept_id:
                details.append(
                    {
                        "query": q[:120],
                        "deleted": result.deleted,
                        "kept_id": result.kept_id,
                    }
                )

        # Sweep leftover task records with no query metadata
        for item in _list_knowledge_items(self.knowledge):
            meta = item.metadata or {}
            text = item.text or ""
            if meta.get("kind") == "task_record" or text.upper().startswith("TASK_RECORD:"):
                self._delete(item.id)
                total_deleted += 1
                details.append({"query": "", "deleted": 1, "kept_id": "", "id": item.id})

        # Drop agent writebacks that are meta/refusal chrome (never durable knowledge)
        for item in _list_knowledge_items(self.knowledge):
            meta = item.metadata or {}
            is_wb = meta.get("kind") == "agent_writeback" or str(item.id).startswith(
                "learned"
            )
            if not is_wb:
                continue
            if is_low_value(item.text or ""):
                self._delete(item.id)
                total_deleted += 1
                details.append(
                    {
                        "query": str(meta.get("query") or "")[:120],
                        "deleted": 1,
                        "kept_id": "",
                        "id": item.id,
                        "reason": "low_value",
                    }
                )

        remaining = 0
        for item in _list_knowledge_items(self.knowledge):
            meta = item.metadata or {}
            if meta.get("kind") == "agent_writeback" or str(item.id).startswith("learned"):
                remaining += 1

        return CorpusCompactionResult(
            deleted=total_deleted,
            queries_compacted=len(queries),
            remaining_writebacks=remaining,
            details=details,
        )

    def _delete(self, item_id: str) -> None:
        if not item_id:
            return
        delete = getattr(self.knowledge, "delete", None)
        if callable(delete):
            delete(item_id)

    def _get_by_id(self, item_id: str) -> MemoryItem | None:
        # In-memory / OS: search then match id; stores don't all expose get()
        store = self.knowledge
        if hasattr(store, "_items"):
            return getattr(store, "_items").get(item_id)
        hits = self.knowledge.search(item_id, k=5)
        for h in hits:
            if h.id == item_id:
                return h
        return None

    def _find_near_duplicate(
        self,
        text: str,
        *,
        embedding: list[float] | None,
        task: str | None,
        query: str,
    ) -> MemoryItem | None:
        hits = self.knowledge.search(
            text[:400],
            k=8,
            embedding=embedding,
            task=task,
            soft_task=True,
        )
        qn = _normalize(query)
        for hit in hits:
            meta = hit.metadata or {}
            if meta.get("kind") == "task_record":
                continue
            if meta.get("query") and _normalize(str(meta.get("query"))) == qn:
                return hit
            if near_duplicate(
                text,
                hit.text or "",
                emb_a=embedding,
                emb_b=hit.embedding,
                lexical_thresh=self.novelty_overlap,
            ):
                return hit
        return None


def should_ingest_knowledge_hit(
    item: MemoryItem,
    *,
    query: str | None = None,
) -> bool:
    """Filter noise and off-topic docs out of Searcher context ingestion."""
    kind = (item.metadata or {}).get("kind")
    text = item.text or ""
    if kind == "task_record":
        return False
    if text.lstrip().upper().startswith("TASK_RECORD:"):
        return False
    if is_low_value(text):
        return False

    # Agent writebacks are query-scoped. Never feed another query's ANSWER/FACT
    # into this prompt (common with weak/mock embeddings + OpenSearch hybrid).
    is_wb = kind == "agent_writeback" or str(item.id).startswith("learned")
    if is_wb and query:
        meta_q = str((item.metadata or {}).get("query") or "").strip()
        if meta_q:
            if not _queries_align(meta_q, query):
                return False
        else:
            # Legacy writeback: parse "ANSWER (query: …):" / "FACT (query: …):"
            m = re.match(
                r"^(?:answer|fact)\s*\(\s*query:\s*([^)]+)\)\s*:",
                text,
                re.I,
            )
            if m and not _queries_align(m.group(1), query):
                return False
            if not m and not _queries_align(text[:160], query):
                # No query tag and body unrelated to current prompt — skip
                return False

    # Seeded corpus (and anything else) must still be about this query.
    # Without this, Acme facts leak into "moons of Jupiter" via weak hybrid hits.
    if query and (query or "").strip():
        if not _item_matches_query(text, query):
            return False
    return True


def _item_matches_query(text: str, query: str, *, min_overlap: float = 0.18) -> bool:
    """True when memory/knowledge text shares substantive terms with the query."""
    if _queries_align(text[:240], query, min_overlap=0.35):
        return True
    return _lexical_overlap(text, query) >= min_overlap


def item_matches_query(text: str, query: str, *, min_overlap: float = 0.18) -> bool:
    """Public alias for query-relevance gating."""
    return _item_matches_query(text, query, min_overlap=min_overlap)


def _queries_align(a: str, b: str, *, min_overlap: float = 0.45) -> bool:
    """True when two query strings refer to roughly the same ask."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return _lexical_overlap(na, nb) >= min_overlap or _jaccard(na, nb) >= 0.35


def is_cross_query_writeback(item: MemoryItem, query: str) -> bool:
    """True if working-memory text is clearly a writeback for a different query."""
    text = item.text or ""
    meta = item.metadata or {}
    kind = meta.get("kind")
    is_wb = (
        kind == "agent_writeback"
        or str(item.id).startswith("know-learned")
        or str(item.id).startswith("learned")
        or bool(re.match(r"^(?:answer|fact)\s*\(\s*query:", text, re.I))
    )
    if not is_wb:
        return False
    meta_q = str(meta.get("query") or "").strip()
    if meta_q:
        return not _queries_align(meta_q, query)
    m = re.match(r"^(?:answer|fact)\s*\(\s*query:\s*([^)]+)\)\s*:", text, re.I)
    if m:
        return not _queries_align(m.group(1), query)
    return False


def dedupe_texts(items: Iterable[MemoryItem]) -> list[MemoryItem]:
    """Exact-text dedupe preserving score order."""
    seen: set[str] = set()
    out: list[MemoryItem] = []
    for item in items:
        key = canonicalize_fact_text(item.text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def dedupe_near(
    items: Iterable[MemoryItem],
    *,
    lexical_thresh: float = 0.55,
) -> list[MemoryItem]:
    """Drop near-duplicates; keep higher-scored / earlier items."""
    kept: list[MemoryItem] = []
    for item in items:
        if is_low_value(item.text or ""):
            continue
        if any(
            near_duplicate(
                item.text or "",
                other.text or "",
                emb_a=item.embedding,
                emb_b=other.embedding,
                lexical_thresh=lexical_thresh,
            )
            for other in kept
        ):
            continue
        kept.append(item)
    return kept


def select_knowledge_for_context(
    hits: Iterable[MemoryItem],
    *,
    query: str | None = None,
    max_items: int = 3,
) -> list[MemoryItem]:
    """Ingest only a compact, non-redundant, query-relevant knowledge slice."""
    filtered = [h for h in hits if should_ingest_knowledge_hit(h, query=query)]
    # Prefer canonical writebacks and seeded docs over verbose paraphrases
    filtered.sort(
        key=lambda h: (
            0 if (h.metadata or {}).get("canonical") else 1,
            0 if (h.metadata or {}).get("kind") == "agent_writeback" else 1,
            -(h.score or 0.0),
            h.token_length or max(1, (len(h.text or "") + 3) // 4),
        )
    )
    return dedupe_near(filtered, lexical_thresh=0.50)[:max_items]


def schedule_compact_after_prompt(
    writeback: KnowledgeWriteback,
    query: str,
    *,
    embedding: list[float] | None = None,
    task: str | None = None,
) -> dict:
    """Run query-slot compaction in a daemon thread after results are returned."""
    import logging
    import threading

    log = logging.getLogger("rcr_router.compact")
    if not (query or "").strip():
        return {"status": "skipped", "reason": "empty_query"}

    def _job() -> None:
        try:
            result = writeback.compact_after_prompt(
                query,
                embedding=embedding,
                task=task,
                force=True,
            )
            log.info(
                "background compact query=%r deleted=%s kept=%s",
                query[:80],
                result.deleted,
                result.kept_id,
            )
        except Exception:  # noqa: BLE001
            log.exception("background compact failed for query=%r", query[:80])

    thread = threading.Thread(
        target=_job,
        name="rcr-compact-after-prompt",
        daemon=True,
    )
    thread.start()
    return {"status": "scheduled", "deleted": 0, "kept_id": "", "deleted_ids": []}


def _list_knowledge_items(knowledge: KnowledgeStore) -> list[MemoryItem]:
    """Best-effort scan of knowledge docs (in-memory or OpenSearch)."""
    if hasattr(knowledge, "_items"):
        return list(getattr(knowledge, "_items").values())

    client = getattr(knowledge, "client", None)
    index = getattr(knowledge, "index", None)
    if client is None or not index:
        return []

    try:
        resp = client.search(index=index, body={"size": 500, "query": {"match_all": {}}})
    except Exception:
        return []

    out: list[MemoryItem] = []
    for hit in resp.get("hits", {}).get("hits", []):
        src = dict(hit.get("_source") or {})
        src["id"] = hit.get("_id") or src.get("id") or ""
        try:
            out.append(MemoryItem.from_dict(src))
        except Exception:
            continue
    return out
