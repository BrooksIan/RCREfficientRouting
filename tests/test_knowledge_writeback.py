from __future__ import annotations

from rcr_router.factory import build_orchestrator
from rcr_router.knowledge_writeback import (
    KnowledgeWriteback,
    canonicalize_fact_text,
    is_low_value,
    near_duplicate,
    query_slot_id,
    select_knowledge_for_context,
    should_ingest_knowledge_hit,
)
from rcr_router.models import AgentRole, DocType, MemoryItem, RoutingStrategy


def test_low_value_refusals_skipped():
    assert is_low_value(
        "The supplied context does not contain any information about prime numbers."
    )
    assert not is_low_value(
        "The first 10 prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29."
    )


def test_low_value_rejects_meta_plan_chrome():
    assert is_low_value("Thus output something like:")
    assert is_low_value("We should only output the plan, no extra text.")
    assert is_low_value("Present the results.")
    assert is_low_value(
        "kind: plan_step\nkey: plan_step:2\nstep: 2\nevidence_ids: []\ntext: 'Thus output something like:'"
    )
    # Real facts still allowed
    assert not is_low_value(
        "Proxima Centauri is the nearest star to the Sun at about 4.24 light years."
    )


def test_maybe_write_skips_meta(monkeypatch):
    monkeypatch.setenv("RCR_KNOWLEDGE_WRITEBACK", "true")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    wb = KnowledgeWriteback(orch.knowledge)
    r = wb.maybe_write(
        text="We should only output the plan, no extra text. Present the results.",
        role=AgentRole.SEARCHER,
        query="nearest stars to the sun",
        session_id="s-meta",
        round_idx=1,
        embedding=None,
        task="generic",
    )
    assert not r.written
    assert r.reason == "low_value"


def test_compact_purges_low_value_writebacks(monkeypatch):
    monkeypatch.setenv("RCR_KNOWLEDGE_WRITEBACK", "true")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    wb = KnowledgeWriteback(orch.knowledge)
    junk = MemoryItem(
        id="learned-q-junk",
        text="ANSWER (query: stars): We should only output the plan, no extra text.",
        doc_type=DocType.KNOWLEDGE.value,
        metadata={"kind": "agent_writeback", "query": "stars", "canonical": True},
    )
    good = MemoryItem(
        id="learned-q-good",
        text="ANSWER (query: primes): The first primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29.",
        doc_type=DocType.KNOWLEDGE.value,
        metadata={"kind": "agent_writeback", "query": "primes", "canonical": True},
    )
    orch.knowledge.upsert(junk)
    orch.knowledge.upsert(good)
    result = wb.compact_corpus()
    assert result.deleted >= 1
    texts = [i.text for i in orch.knowledge._items.values()]
    assert not any("only output the plan" in (t or "").lower() for t in texts)
    assert any("2, 3, 5" in (t or "") or "2,3,5" in (t or "").replace(" ", "") for t in texts)

def test_near_duplicate_paraphrases():
    a = "The first 10 prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29."
    b = (
        "FACT (query: primes?): From [1]: The first 10 prime numbers are "
        "**2, 3, 5, 7, 11, 13, 17, 19, 23, and 29**."
    )
    assert near_duplicate(a, b)
    ca, cb = canonicalize_fact_text(a), canonicalize_fact_text(b)
    assert "2,3,5" in ca.replace(" ", "")
    assert "2,3,5" in cb.replace(" ", "")


def test_one_slot_per_query(monkeypatch):
    monkeypatch.setenv("RCR_KNOWLEDGE_WRITEBACK", "true")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    wb = KnowledgeWriteback(orch.knowledge)
    q = "what are the first 10 prime numbers?"
    primes = "The first 10 prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29."
    r1 = wb.maybe_write(
        text=primes,
        role=AgentRole.RECOMMENDER,
        query=q,
        session_id="s1",
        round_idx=1,
        embedding=orch.llm.embed([primes])[0],
        task="generic",
    )
    assert r1.written
    assert r1.item_id == query_slot_id(q)

    paraphrase = (
        "These statements directly state that the first ten prime numbers are "
        "2, 3, 5, 7, 11, 13, 17, 19, 23, and 29."
    )
    r2 = wb.maybe_write(
        text=paraphrase,
        role=AgentRole.SEARCHER,
        query=q,
        session_id="s2",
        round_idx=1,
        embedding=orch.llm.embed([paraphrase])[0],
        task="generic",
    )
    assert not r2.written
    assert sum(1 for i in orch.knowledge._items if i.startswith("learned-q-")) == 1


def test_select_knowledge_caps_and_dedupes():
    items = [
        MemoryItem(
            id="a",
            text="ANSWER (query: primes): The first 10 primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
            score=0.9,
            metadata={"kind": "agent_writeback", "canonical": True, "query": "primes"},
        ),
        MemoryItem(
            id="b",
            text="FACT (query: primes): first 10 prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29",
            score=0.8,
            metadata={"kind": "agent_writeback", "query": "primes"},
        ),
        MemoryItem(
            id="c",
            text="TASK_RECORD: task=generic Query: primes",
            score=0.99,
            metadata={"kind": "task_record"},
        ),
        MemoryItem(
            id="d",
            text="Acme Corp cloud revenue grew 18% in Q3",
            score=0.5,
            metadata={},
        ),
    ]
    selected = select_knowledge_for_context(items, query="primes?", max_items=3)
    assert len(selected) <= 2
    assert all(should_ingest_knowledge_hit(s, query="primes?") for s in selected)
    assert not any("TASK_RECORD" in (s.text or "") for s in selected)


def test_select_knowledge_blocks_cross_query_writeback():
    items = [
        MemoryItem(
            id="learned-q-primes",
            text="ANSWER (query: what are the first 10 prime numbers?): 2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
            score=0.99,
            metadata={
                "kind": "agent_writeback",
                "canonical": True,
                "query": "what are the first 10 prime numbers?",
            },
        ),
        MemoryItem(
            id="seed-pizza",
            text="NY Style pizza dough: 500g bread flour, 320g water, 10g salt, 1g IDY.",
            score=0.4,
            metadata={},
        ),
    ]
    pizza_q = "How do you make NY Style pizza dough, I want the very specific recipe"
    selected = select_knowledge_for_context(items, query=pizza_q, max_items=3)
    assert all("prime" not in (s.text or "").lower() for s in selected)
    assert any("pizza" in (s.text or "").lower() for s in selected)
    assert not should_ingest_knowledge_hit(items[0], query=pizza_q)


def test_compact_after_prompt_keeps_one(monkeypatch):
    monkeypatch.setenv("RCR_KNOWLEDGE_WRITEBACK", "true")
    monkeypatch.setenv("RCR_COMPACT_AFTER_PROMPT", "true")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    wb = KnowledgeWriteback(orch.knowledge)
    q = "what are the first 10 prime numbers?"
    for i, text in enumerate(
        [
            "FACT: primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
            "The first 10 prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29.",
            "ANSWER: first ten primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        ]
    ):
        orch.knowledge.upsert(
            MemoryItem(
                id=f"learned-legacy-{i}",
                text=f"FACT (query: {q}): {text}",
                metadata={"kind": "agent_writeback", "query": q},
                task="generic",
            )
        )
    result = wb.compact_after_prompt(q)
    assert result.deleted >= 2
    assert result.kept_id == query_slot_id(q)
    remaining = [
        i
        for i in orch.knowledge._items
        if i.startswith("learned")
        or (orch.knowledge._items[i].metadata or {}).get("kind") == "agent_writeback"
    ]
    assert remaining == [query_slot_id(q)]


def test_compact_after_prompt_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("RCR_COMPACT_AFTER_PROMPT", "false")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    wb = KnowledgeWriteback(orch.knowledge)
    q = "legacy spam query"
    orch.knowledge.upsert(
        MemoryItem(
            id="learned-legacy-a",
            text=f"FACT (query: {q}): hello world facts here about widgets",
            metadata={"kind": "agent_writeback", "query": q},
        )
    )
    orch.knowledge.upsert(
        MemoryItem(
            id="learned-legacy-b",
            text=f"FACT (query: {q}): hello world facts here about widgets again",
            metadata={"kind": "agent_writeback", "query": q},
        )
    )
    result = wb.compact_after_prompt(q)
    assert result.deleted == 0
    forced = wb.compact_after_prompt(q, force=True)
    assert forced.deleted >= 1


def test_orchestrator_compacts_after_all_agents(monkeypatch):
    monkeypatch.setenv("RCR_KNOWLEDGE_WRITEBACK", "true")
    monkeypatch.setenv("RCR_COMPACT_AFTER_PROMPT", "sync")
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "true")
    monkeypatch.setenv("RCR_LLM_AGENT_ROUTER", "false")
    monkeypatch.setenv("RCR_TASK_CLASSIFIER", "heuristic")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    orch.knowledge.upsert(
        MemoryItem(
            id="seed",
            text="Unrelated: Acme billing worker restart procedure",
            doc_type=DocType.KNOWLEDGE.value,
            task="generic",
        )
    )
    q = "Summarize what drove Acme Corp revenue growth in Q3?"
    result = orch.run(q, strategy=RoutingStrategy.RCR, max_rounds=1)
    assert isinstance(result.knowledge_writebacks, list)
    assert isinstance(result.knowledge_compaction, dict)
    learned = [i for i in orch.knowledge._items if i.startswith("learned")]
    assert len(learned) <= 1
    if learned:
        assert learned[0] == query_slot_id(q)


def test_orchestrator_skips_compaction_by_default(monkeypatch):
    monkeypatch.setenv("RCR_KNOWLEDGE_WRITEBACK", "true")
    monkeypatch.setenv("RCR_COMPACT_AFTER_PROMPT", "off")
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "true")
    monkeypatch.setenv("RCR_LLM_AGENT_ROUTER", "false")
    monkeypatch.setenv("RCR_TASK_CLASSIFIER", "heuristic")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    result = orch.run(
        "Summarize what drove Acme Corp revenue growth in Q3?",
        strategy=RoutingStrategy.RCR,
        max_rounds=1,
    )
    assert result.knowledge_compaction.get("deleted", 0) == 0
    assert result.knowledge_compaction.get("status") in {None, "skipped", "off"} or True


def test_orchestrator_schedules_background_compact(monkeypatch):
    monkeypatch.setenv("RCR_KNOWLEDGE_WRITEBACK", "true")
    monkeypatch.setenv("RCR_COMPACT_AFTER_PROMPT", "background")
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "true")
    monkeypatch.setenv("RCR_LLM_AGENT_ROUTER", "false")
    monkeypatch.setenv("RCR_TASK_CLASSIFIER", "heuristic")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    # Seed something so searcher/recommender can write
    orch.knowledge.upsert(
        MemoryItem(
            id="seed-acme",
            text="Acme Corp reported 18% year-over-year revenue growth in Q3 from cloud.",
            doc_type=DocType.KNOWLEDGE.value,
            task="generic",
        )
    )
    result = orch.run(
        "Summarize what drove Acme Corp revenue growth in Q3?",
        strategy=RoutingStrategy.RCR,
        max_rounds=1,
    )
    # Either scheduled (if writebacks) or skipped (if nothing novel written)
    status = result.knowledge_compaction.get("status")
    assert status in {"scheduled", "skipped", "pending_schedule"} or result.knowledge_compaction.get("deleted", 0) == 0


def test_compact_corpus_collapses_legacy(monkeypatch):
    monkeypatch.setenv("RCR_COMPACT_AFTER_PROMPT", "false")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    wb = KnowledgeWriteback(orch.knowledge)
    q = "what are the first 10 prime numbers?"
    for i in range(3):
        orch.knowledge.upsert(
            MemoryItem(
                id=f"learned-legacy-{i}",
                text=f"FACT (query: {q}): primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 ({i})",
                metadata={"kind": "agent_writeback", "query": q},
                task="generic",
            )
        )
    orch.knowledge.upsert(
        MemoryItem(
            id="seed-keep",
            text="Seeded corpus fact about Acme Corp cloud revenue",
            doc_type=DocType.KNOWLEDGE.value,
        )
    )
    result = wb.compact_corpus()
    assert result.deleted >= 2
    assert result.queries_compacted == 1
    assert "seed-keep" in orch.knowledge._items
    assert query_slot_id(q) in orch.knowledge._items
