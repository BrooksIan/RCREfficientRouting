from __future__ import annotations

from .budget import TokenBudgetAllocator
from .knowledge_writeback import (
    dedupe_near,
    dedupe_texts,
    is_cross_query_writeback,
    item_matches_query,
    near_duplicate,
    route_near_dedupe_enabled,
)
from .models import (
    STAGE_FOR_ROLE,
    AgentRole,
    DocType,
    MemoryItem,
    RoutingStrategy,
    TaskStage,
)
from .scorer import ImportanceScorer
from .store import MemoryStore

# Soft cap on plan_step units routed into any agent prompt.
_MAX_PLAN_STEPS = 4


def _is_task_record(item: MemoryItem) -> bool:
    kind = (item.metadata or {}).get("kind")
    if kind == "task_record":
        return True
    return (item.text or "").lstrip().upper().startswith("TASK_RECORD:")


def _is_plan_step(item: MemoryItem) -> bool:
    kind = item.struct_kind or (item.metadata or {}).get("struct_kind")
    if kind == "plan_step":
        return True
    return item.doc_type == DocType.PLAN.value and "plan_step:" in (
        item.struct_key or (item.metadata or {}).get("struct_key") or ""
    )


def _is_user_query(item: MemoryItem) -> bool:
    return (item.text or "").lstrip().upper().startswith("USER_QUERY:")


def _dedupe_for_route(items: list[MemoryItem]) -> list[MemoryItem]:
    """Exact-text dedupe; always near-dedupe plan steps; optional full near-dedupe."""
    if route_near_dedupe_enabled():
        return dedupe_near(items, lexical_thresh=0.55)
    # Always collapse paraphrased plan_step spam (common across rounds).
    plans: list[MemoryItem] = []
    others: list[MemoryItem] = []
    for item in items:
        if _is_plan_step(item):
            plans.append(item)
        else:
            others.append(item)
    plans = dedupe_near(plans, lexical_thresh=0.45)[:_MAX_PLAN_STEPS]
    merged = dedupe_texts(others + plans)
    # Preserve original score order
    order = {id(i): n for n, i in enumerate(items)}
    merged.sort(key=lambda m: order.get(id(m), 10_000))
    return merged


def _filter_route_items(items: list[MemoryItem], query: str) -> list[MemoryItem]:
    """Drop task records, cross-query writebacks, and off-topic corpus noise."""
    out: list[MemoryItem] = []
    for i in items:
        if _is_task_record(i) or is_cross_query_writeback(i, query):
            continue
        if _is_user_query(i):
            out.append(i)
            continue
        # Keep structured session artifacts even if terse; gate free-text knowledge.
        if i.doc_type == DocType.KNOWLEDGE.value or i.source_agent == "knowledge":
            if query and not item_matches_query(i.text or "", query):
                continue
        out.append(i)
    return out


def _is_evidence(item: MemoryItem) -> bool:
    if _is_user_query(item):
        return False
    if item.doc_type in {DocType.FACT.value, DocType.KNOWLEDGE.value, DocType.PLAN.value}:
        return True
    if item.source_agent in {"searcher", "knowledge", "planner"}:
        return True
    if item.role_tag in {AgentRole.SEARCHER.value, AgentRole.PLANNER.value}:
        return True
    return False


def _pin_for_recommender(items: list[MemoryItem]) -> list[MemoryItem]:
    """Ensure user query + evidence precede weaker items for the Recommender."""
    pinned: list[MemoryItem] = []
    rest: list[MemoryItem] = []
    seen: set[str] = set()
    for item in items:
        if _is_user_query(item) and item.id not in seen:
            pinned.append(item)
            seen.add(item.id)
    for item in items:
        if item.id in seen:
            continue
        if _is_evidence(item):
            pinned.append(item)
            seen.add(item.id)
        else:
            rest.append(item)
    return pinned + rest


class RCRRouter:
    """Algorithm 1: score → sort → greedy fill under B_i."""

    def __init__(
        self,
        memory: MemoryStore,
        budget: TokenBudgetAllocator | None = None,
        scorer: ImportanceScorer | None = None,
        full_context_cap: int | None = None,
    ) -> None:
        self.memory = memory
        self.budget = budget or TokenBudgetAllocator.from_env()
        self.scorer = scorer or ImportanceScorer()
        self.full_context_cap = (
            full_context_cap
            if full_context_cap is not None
            else self.budget.full_context_cap
        )

    def route(
        self,
        *,
        query: str,
        role: AgentRole,
        session_id: str,
        strategy: RoutingStrategy = RoutingStrategy.RCR,
        stage: TaskStage | None = None,
        embedding: list[float] | None = None,
        query_task: str | None = None,
        k: int = 50,
    ) -> list[MemoryItem]:
        stage = stage or STAGE_FOR_ROLE[role]
        budget_tokens = self.budget.budget_for(role)

        if strategy == RoutingStrategy.FULL:
            items = _filter_route_items(self.memory.list_session(session_id), query)
            for item in items:
                item.score = 1.0
            ordered = _dedupe_for_route(items)
            if role == AgentRole.RECOMMENDER:
                ordered = _pin_for_recommender(ordered)
            return self._fill(ordered, budget_tokens=self.full_context_cap)

        if strategy == RoutingStrategy.STATIC:
            items = self.memory.search(
                query,
                session_id=session_id,
                role_tag=role.value,
                task=query_task,
                k=k,
                embedding=None,
            )
            if not items:
                items = self.memory.search(
                    query,
                    session_id=session_id,
                    stage_tag=stage.value,
                    task=query_task,
                    k=k,
                    embedding=None,
                )
            # Static recommender often sees almost nothing (role_tag=recommender rare).
            # Pull planner/searcher stage items so it can still answer.
            if role == AgentRole.RECOMMENDER and len(items) < 3:
                extra = self.memory.search(
                    query,
                    session_id=session_id,
                    k=k,
                    embedding=None,
                )
                seen = {i.id for i in items}
                for e in extra:
                    if e.id not in seen:
                        items.append(e)
                        seen.add(e.id)
            items = _filter_route_items(items, query)
            for item in items:
                item.score = 1.0 if item.role_tag == role.value else 0.5
            ordered = _dedupe_for_route(items)
            if role == AgentRole.RECOMMENDER:
                ordered = _pin_for_recommender(ordered)
            return self._fill(ordered, budget_tokens=budget_tokens)

        candidates = self.memory.search(
            query,
            session_id=session_id,
            task=query_task,
            k=k,
            embedding=embedding,
        )
        candidates = _filter_route_items(candidates, query)
        scored = self.scorer.score_many(
            candidates,
            role,
            stage,
            query_embedding=embedding,
            query_task=query_task,
        )
        scored.sort(key=lambda m: m.score, reverse=True)
        scored = _dedupe_for_route(scored)
        if role == AgentRole.RECOMMENDER:
            scored = _pin_for_recommender(scored)
        elif role == AgentRole.PLANNER:
            scored = _prefer_facts_over_extra_plans(scored)
        return self._fill(scored, budget_tokens=budget_tokens)

    def _fill(self, items: list[MemoryItem], *, budget_tokens: int) -> list[MemoryItem]:
        selected: list[MemoryItem] = []
        total = 0
        plan_count = 0
        for item in items:
            if _is_plan_step(item):
                if plan_count >= _MAX_PLAN_STEPS:
                    continue
                # Skip near-dupes of plans already selected
                if any(
                    near_duplicate(
                        item.text or "",
                        other.text or "",
                        lexical_thresh=0.45,
                    )
                    for other in selected
                    if _is_plan_step(other)
                ):
                    continue
            tl = item.token_length or 1
            if total + tl > budget_tokens:
                # Still allow a single oversized pinned item if nothing selected yet
                if not selected and tl <= budget_tokens * 2:
                    selected.append(item)
                    if _is_plan_step(item):
                        plan_count += 1
                break
            selected.append(item)
            if _is_plan_step(item):
                plan_count += 1
            total += tl
        return selected


def _prefer_facts_over_extra_plans(items: list[MemoryItem]) -> list[MemoryItem]:
    """Planner: surface facts/answers before a long tail of plan paraphrases."""
    head: list[MemoryItem] = []
    facts: list[MemoryItem] = []
    plans: list[MemoryItem] = []
    rest: list[MemoryItem] = []
    for item in items:
        if _is_user_query(item):
            head.append(item)
        elif item.struct_kind in {"fact", "evidence", "answer"} or item.doc_type in {
            DocType.FACT.value,
            DocType.KNOWLEDGE.value,
        }:
            facts.append(item)
        elif _is_plan_step(item):
            plans.append(item)
        else:
            rest.append(item)
    return head + facts + plans[:_MAX_PLAN_STEPS] + rest
