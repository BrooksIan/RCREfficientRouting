"""Routing hygiene: off-topic knowledge + plan-step spam."""

from __future__ import annotations

from rcr_router.budget import TokenBudgetAllocator
from rcr_router.knowledge_writeback import should_ingest_knowledge_hit
from rcr_router.memory_store_memory import InMemoryMemoryStore
from rcr_router.models import AgentRole, DocType, MemoryItem, RoutingStrategy, TaskStage
from rcr_router.router import RCRRouter
from rcr_router.structured_memory import extract_structured_units


def test_seeded_acme_blocked_for_jupiter_query():
    item = MemoryItem(
        text="Management guided that cloud will remain the primary growth engine for Acme Corp.",
        doc_type=DocType.KNOWLEDGE.value,
        source_agent="knowledge",
    )
    assert not should_ingest_knowledge_hit(
        item, query="name 5 largest moons of Jupiter"
    )


def test_jupiter_fact_allowed_for_jupiter_query():
    item = MemoryItem(
        text="Ganymede, Callisto, Io, and Europa are the four largest moons of Jupiter.",
        doc_type=DocType.KNOWLEDGE.value,
        source_agent="knowledge",
    )
    assert should_ingest_knowledge_hit(item, query="name 5 largest moons of Jupiter")


def test_plan_extract_dedupes_paraphrases():
    text = (
        "Plan:\n"
        "Search for the largest moons of Jupiter.\n"
        "Retrieve their names.\n"
        "Identify the largest moons of Jupiter by size or diameter.\n"
        "Find information about moons of Jupiter.\n"
    )
    units = extract_structured_units(text, role=AgentRole.PLANNER)
    assert 1 <= len(units) <= 3
    assert all(u.kind == "plan_step" for u in units)


def test_router_drops_offtopic_knowledge_and_caps_plans():
    mem = InMemoryMemoryStore()
    session = "jupiter-hygiene"
    query = "name 5 largest moons of Jupiter"
    mem.upsert(
        MemoryItem(
            text=f"USER_QUERY: {query}",
            role_tag="user",
            session_id=session,
            doc_type=DocType.INTERACTION.value,
        )
    )
    mem.upsert(
        MemoryItem(
            text="Management guided that cloud will remain the primary growth engine for Acme Corp.",
            role_tag="searcher",
            source_agent="knowledge",
            session_id=session,
            doc_type=DocType.KNOWLEDGE.value,
            score=0.9,
        )
    )
    mem.upsert(
        MemoryItem(
            text="kind: fact\nkey: fact:moons\ntext: Ganymede Callisto Io Europa Amalthea\n",
            role_tag="searcher",
            source_agent="searcher",
            session_id=session,
            doc_type=DocType.FACT.value,
            struct_kind="fact",
            struct_key="fact:moons",
            score=0.8,
        )
    )
    for i, step in enumerate(
        [
            "Search for the largest moons of Jupiter",
            "Find information about moons of Jupiter",
            "Retrieve names of the largest moons of Jupiter",
            "Identify the largest moons of Jupiter by size",
            "Look up Jupiter moon diameters",
            "Get moon list for Jupiter",
        ],
        start=1,
    ):
        mem.upsert(
            MemoryItem(
                text=f"{i}. {step}",
                role_tag="planner",
                source_agent="planner",
                session_id=session,
                doc_type=DocType.PLAN.value,
                struct_kind="plan_step",
                struct_key=f"plan_step:{i}",
                structure={"step": i},
                score=0.5,
            )
        )

    router = RCRRouter(
        mem,
        budget=TokenBudgetAllocator(preset="savings"),
    )
    routed = router.route(
        query=query,
        role=AgentRole.PLANNER,
        session_id=session,
        strategy=RoutingStrategy.RCR,
        stage=TaskStage.PLANNING,
    )
    texts = " | ".join(i.text for i in routed)
    assert "Acme" not in texts
    assert "Ganymede" in texts or "fact" in texts.lower()
    plan_n = sum(1 for i in routed if (i.struct_kind or "") == "plan_step")
    assert plan_n <= 4
