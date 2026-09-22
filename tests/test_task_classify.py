from __future__ import annotations

from rcr_router.factory import build_orchestrator
from rcr_router.models import DocType, MemoryItem, RoutingStrategy
from rcr_router.task_classify import (
    TaskClassifier,
    heuristic_classify,
    task_match_score,
)


def test_heuristic_classify_synthesis():
    d = heuristic_classify("What drove Acme Corp revenue growth in Q3? Summarize the key takeaway.")
    assert d.task == "synthesis"
    assert d.confidence > 0.4


def test_heuristic_classify_coding():
    d = heuristic_classify("Please implement a Python function to parse stack traces and refactor the API endpoint.")
    assert d.task == "coding"


def test_task_match_score():
    assert task_match_score("coding", "coding") == 1.0
    assert task_match_score("extraction", "coding") < 0.5
    assert task_match_score(None, "generic") == 0.5


def test_knowledge_search_prefers_matching_task():
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    orch.knowledge.upsert(
        MemoryItem(
            id="k-code",
            text="How to fix a Python TypeError in unit tests",
            doc_type=DocType.KNOWLEDGE.value,
            task="coding",
        )
    )
    orch.knowledge.upsert(
        MemoryItem(
            id="k-synth",
            text="Acme Corp Q3 revenue grew on cloud subscriptions",
            doc_type=DocType.KNOWLEDGE.value,
            task="synthesis",
        )
    )
    hits = orch.knowledge.search(
        "Acme revenue cloud",
        k=2,
        task="synthesis",
        soft_task=True,
    )
    assert hits
    assert hits[0].task == "synthesis"


def test_orchestrator_classifies_and_persists_task(monkeypatch):
    monkeypatch.setenv("RCR_TASK_CLASSIFIER", "heuristic")
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "true")
    monkeypatch.setenv("RCR_LLM_AGENT_ROUTER", "false")
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    orch.knowledge.upsert(
        MemoryItem(
            id="k1",
            text="Acme Corp cloud revenue grew in Q3 primarily from subscriptions",
            doc_type=DocType.KNOWLEDGE.value,
            task="synthesis",
        )
    )
    result = orch.run(
        "Summarize what drove Acme Corp revenue growth in Q3?",
        strategy=RoutingStrategy.RCR,
        max_rounds=1,
    )
    assert result.task.get("task") == "synthesis"
    # User query memory tagged
    user_items = [m for m in result.memory_snapshot if m.get("role_tag") == "user"]
    assert user_items and user_items[0].get("task") == "synthesis"
    # Task classification is on the user item / RunResult — not a TASK_RECORD doc
    assert user_items[0].get("metadata", {}).get("task") == "synthesis"


def test_llm_classifier_mode(monkeypatch):
    monkeypatch.setenv("RCR_TASK_CLASSIFIER", "llm")
    from rcr_router.llm import MockLLMClient

    clf = TaskClassifier(llm=MockLLMClient(), mode="llm")
    d = clf.classify("Extract factual claims from the passage about Acme")
    assert d.task == "extraction"
    assert d.source == "llm"
