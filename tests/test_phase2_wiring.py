from __future__ import annotations

import json
from pathlib import Path

from rcr_router.opensearch_store import load_index_template
from rcr_router.llm import MockLLMClient, build_llm_client
from rcr_router.factory import build_orchestrator
from rcr_router.models import RoutingStrategy


def test_index_template_loads():
    tmpl = load_index_template()
    assert "rcr-working-memory" in tmpl["index_patterns"]
    assert tmpl["template"]["mappings"]["properties"]["embedding"]["type"] == "knn_vector"


def test_build_llm_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "true")
    client = build_llm_client()
    assert isinstance(client, MockLLMClient)


def test_smoke_t1_and_t3(monkeypatch):
    monkeypatch.setenv("RCR_USE_OPENSEARCH", "false")
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "true")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    corpus = json.loads(
        (Path(__file__).resolve().parents[1] / "assets" / "data" / "knowledge-corpus.json").read_text()
    )
    from rcr_router.models import MemoryItem, DocType

    embs = orch.llm.embed([r["text"] for r in corpus])
    for row, emb in zip(corpus, embs):
        orch.knowledge.upsert(
            MemoryItem(id=row["id"], text=row["text"], doc_type=DocType.KNOWLEDGE.value, embedding=emb)
        )
    r1 = orch.run("Acme Corp Q3 revenue growth?", strategy=RoutingStrategy.RCR, max_rounds=1)
    r3 = orch.run("Acme Corp Q3 revenue growth?", strategy=RoutingStrategy.RCR, max_rounds=3)
    assert r1.usage["calls"] == 3
    assert r3.usage["calls"] == 9
    assert r1.answer and r3.answer
