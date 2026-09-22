from __future__ import annotations

from rcr_router.budget import TokenBudgetAllocator
from rcr_router.models import AgentRole, MemoryItem, RoutingStrategy, TaskStage
from rcr_router.router import RCRRouter
from rcr_router.scorer import ImportanceScorer
from rcr_router.memory_store_memory import InMemoryKnowledgeStore, InMemoryMemoryStore
from rcr_router.memory_update import MemoryUpdater


def test_budget_role_offsets():
    alloc = TokenBudgetAllocator(base=200)
    assert alloc.budget_for(AgentRole.PLANNER) > alloc.budget_for(AgentRole.RECOMMENDER)
    assert alloc.budget_for(AgentRole.RECOMMENDER) >= alloc.budget_for(AgentRole.SEARCHER)


def test_router_never_exceeds_budget():
    mem = InMemoryMemoryStore()
    session = "s1"
    # Many small items that would exceed a tight budget if all taken
    for i in range(30):
        mem.upsert(
            MemoryItem(
                text=f"plan evidence fact document answer item {i} " + ("word " * 10),
                session_id=session,
                role_tag=AgentRole.PLANNER.value if i % 2 == 0 else AgentRole.SEARCHER.value,
                stage_tag=TaskStage.PLANNING.value,
            )
        )
    router = RCRRouter(mem, budget=TokenBudgetAllocator(base=50, role_offsets={
        AgentRole.PLANNER: 0,
        AgentRole.SEARCHER: 0,
        AgentRole.RECOMMENDER: 0,
    }))
    selected = router.route(
        query="plan evidence",
        role=AgentRole.PLANNER,
        session_id=session,
        strategy=RoutingStrategy.RCR,
    )
    total = sum(i.token_length for i in selected)
    assert total <= 50
    assert selected  # got something


def test_static_vs_rcr_differ():
    mem = InMemoryMemoryStore()
    session = "s2"
    mem.upsert(
        MemoryItem(
            text="planner plan decompose strategy for revenue question",
            session_id=session,
            role_tag=AgentRole.PLANNER.value,
            stage_tag=TaskStage.PLANNING.value,
        )
    )
    mem.upsert(
        MemoryItem(
            text="unrelated cooking recipe pasta tomatoes",
            session_id=session,
            role_tag=AgentRole.SEARCHER.value,
            stage_tag=TaskStage.SEARCHING.value,
        )
    )
    mem.upsert(
        MemoryItem(
            text="evidence fact retrieve document about revenue metrics",
            session_id=session,
            role_tag=AgentRole.SEARCHER.value,
            stage_tag=TaskStage.SEARCHING.value,
        )
    )
    router = RCRRouter(mem)
    static = router.route(
        query="revenue metrics",
        role=AgentRole.SEARCHER,
        session_id=session,
        strategy=RoutingStrategy.STATIC,
    )
    rcr = router.route(
        query="revenue metrics",
        role=AgentRole.SEARCHER,
        session_id=session,
        strategy=RoutingStrategy.RCR,
    )
    static_ids = {i.id for i in static}
    rcr_ids = {i.id for i in rcr}
    # Strategies can overlap but selection ordering/scoring paths differ;
    # at minimum static is constrained to searcher-tagged items.
    assert all(i.role_tag == AgentRole.SEARCHER.value for i in static)
    assert rcr_ids  # rcr returns hits
    assert static_ids != rcr_ids or [i.score for i in static] != [i.score for i in rcr]


def test_memory_update_filters_duplicates():
    mem = InMemoryMemoryStore()
    updater = MemoryUpdater(mem)
    session = "s3"
    first = updater.update_from_agent_output(
        text="PLAN: investigate quarterly revenue drivers",
        role=AgentRole.PLANNER,
        stage=TaskStage.PLANNING,
        session_id=session,
        round_idx=1,
    )
    assert first
    dup = updater.update_from_agent_output(
        text="PLAN: investigate quarterly revenue drivers",
        role=AgentRole.PLANNER,
        stage=TaskStage.PLANNING,
        session_id=session,
        round_idx=2,
        existing=mem.list_session(session),
    )
    assert dup == []
    assert len(mem.list_session(session)) == 1


def test_scorer_prefers_role_keywords():
    scorer = ImportanceScorer(now=1_700_000_000)
    plan_item = MemoryItem(
        text="create a plan and decompose the query into subquestions",
        timestamp=1_700_000_000,
    )
    other = MemoryItem(text="completely unrelated gardening tips", timestamp=1_700_000_000)
    s1 = scorer.score(plan_item, AgentRole.PLANNER, TaskStage.PLANNING, retrieval_score=0.1)
    s2 = scorer.score(other, AgentRole.PLANNER, TaskStage.PLANNING, retrieval_score=0.1)
    assert s1 > s2
