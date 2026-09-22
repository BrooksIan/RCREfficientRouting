from rcr_router.llm import MockLLMClient
from rcr_router.memory_store_memory import InMemoryKnowledgeStore, InMemoryMemoryStore
from rcr_router.models import DocType, MemoryItem, RoutingStrategy
from rcr_router.orchestrator import Orchestrator


def test_orchestrator_runs_all_roles():
    mem = InMemoryMemoryStore()
    know = InMemoryKnowledgeStore()
    know.upsert(
        MemoryItem(
            text="Acme Corp reported revenue growth in Q3 driven by cloud services.",
            doc_type=DocType.KNOWLEDGE.value,
            embedding=MockLLMClient().embed(["Acme Corp revenue growth cloud"])[0],
        )
    )
    orch = Orchestrator(mem, know, llm=MockLLMClient())
    result = orch.run(
        "What drove Acme Corp revenue growth in Q3?",
        strategy=RoutingStrategy.RCR,
        max_rounds=2,
    )
    assert result.answer
    assert result.usage["calls"] == 2 * 3  # rounds * roles
    roles_seen = {t.role for t in result.traces}
    assert roles_seen == {"planner", "searcher", "recommender"}
    assert any(m.get("doc_type") == DocType.KNOWLEDGE.value for m in result.memory_snapshot)
