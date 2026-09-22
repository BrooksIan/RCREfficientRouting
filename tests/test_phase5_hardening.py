from __future__ import annotations

import os
import uuid

import pytest

from rcr_router.llm import MockLLMClient
from rcr_router.models import DocType, MemoryItem, RoutingStrategy, TaskStage, AgentRole
from rcr_router.opensearch_store import (
    OpenSearchKnowledgeStore,
    OpenSearchMemoryStore,
    ensure_indices,
    opensearch_available,
    opensearch_client_from_env,
)
from rcr_router.orchestrator import Orchestrator
from rcr_router.scorer import ImportanceScorer


pytestmark = pytest.mark.skipif(
    not opensearch_available(),
    reason="OpenSearch not reachable on OPENSEARCH_HOST:PORT",
)


@pytest.fixture()
def os_client():
    os.environ.setdefault("OPENSEARCH_EMBED_DIM", "8")
    client = opensearch_client_from_env()
    ensure_indices(client, embed_dim=8)
    return client


def test_opensearch_memory_roundtrip(os_client):
    mem = OpenSearchMemoryStore(os_client)
    session = f"test-{uuid.uuid4()}"
    llm = MockLLMClient(dim=8)
    text = "planner plan decompose strategy for cloud revenue"
    emb = llm.embed([text])[0]
    item = MemoryItem(
        text=text,
        session_id=session,
        role_tag=AgentRole.PLANNER.value,
        stage_tag=TaskStage.PLANNING.value,
        embedding=emb,
        doc_type=DocType.PLAN.value,
    )
    mem.upsert(item)
    listed = mem.list_session(session)
    assert any(i.id == item.id for i in listed)
    hits = mem.search("cloud revenue", session_id=session, k=5, embedding=emb)
    assert hits
    deleted = mem.delete_session(session)
    assert deleted >= 1
    assert mem.list_session(session) == []


def test_opensearch_orchestrator_smoke(os_client):
    llm = MockLLMClient(dim=8)
    mem = OpenSearchMemoryStore(os_client)
    know = OpenSearchKnowledgeStore(os_client)
    know.upsert(
        MemoryItem(
            id=f"know-{uuid.uuid4()}",
            text="Acme Corp reported cloud-driven revenue growth in Q3.",
            doc_type=DocType.KNOWLEDGE.value,
            embedding=llm.embed(["Acme Corp cloud revenue growth Q3"])[0],
        )
    )
    orch = Orchestrator(mem, know, llm=llm)
    result = orch.run(
        "What drove Acme Corp revenue growth in Q3?",
        strategy=RoutingStrategy.RCR,
        max_rounds=1,
    )
    assert result.answer
    assert result.usage["calls"] == 3
    # cleanup
    mem.delete_session(result.session_id)


def test_embedding_scorer_boosts_similar_items():
    scorer = ImportanceScorer(now=1_700_000_000)
    llm = MockLLMClient(dim=8)
    query = "Acme Corp Q3 cloud revenue growth"
    q_emb = llm.embed([query])[0]
    relevant = MemoryItem(
        text="Acme Corp cloud revenue grew strongly in Q3",
        embedding=llm.embed(["Acme Corp cloud revenue grew strongly in Q3"])[0],
        timestamp=1_700_000_000,
    )
    noise = MemoryItem(
        text="pasta cafeteria menu seasonal dishes",
        embedding=llm.embed(["pasta cafeteria menu seasonal dishes"])[0],
        timestamp=1_700_000_000,
    )
    s_rel = scorer.score(
        relevant, AgentRole.SEARCHER, TaskStage.SEARCHING, query_embedding=q_emb
    )
    s_noise = scorer.score(
        noise, AgentRole.SEARCHER, TaskStage.SEARCHING, query_embedding=q_emb
    )
    assert s_rel > s_noise
