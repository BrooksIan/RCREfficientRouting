"""Task→model routing + knowledge probe."""

from __future__ import annotations

from rcr_router.endpoint_store import EndpointStoreData, UserEndpoint
from rcr_router.factory import build_orchestrator
from rcr_router.knowledge_probe import (
    KnowledgeProbeResult,
    classify_strength,
    lexical_coverage,
    probe_knowledge,
)
from rcr_router.model_routing import ModelEndpoint, ModelRegistry
from rcr_router.models import DocType, MemoryItem, RoutingStrategy
from rcr_router.task_model_policy import (
    build_run_model_plan,
    load_policy,
    task_model_routing_enabled,
)


def test_routing_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RCR_TASK_MODEL_ROUTING", raising=False)
    assert task_model_routing_enabled() is False
    assert task_model_routing_enabled(False) is False
    assert task_model_routing_enabled(True) is True
    monkeypatch.setenv("RCR_TASK_MODEL_ROUTING", "true")
    assert task_model_routing_enabled() is True


def test_classify_strength_tiers():
    assert classify_strength(hit_count=2, coverage=0.3) == "strong"
    assert classify_strength(hit_count=1, coverage=0.2) == "medium"
    assert classify_strength(hit_count=0, coverage=0.5) == "weak"
    assert classify_strength(hit_count=3, coverage=0.05) == "weak"


def test_lexical_coverage_jupiter():
    cov = lexical_coverage(
        "name 5 largest moons of Jupiter",
        ["Ganymede Callisto Io Europa are large moons of Jupiter"],
    )
    assert cov >= 0.25


def test_probe_knowledge_filters_and_scores():
    from rcr_router.memory_store_memory import InMemoryKnowledgeStore

    ks = InMemoryKnowledgeStore()
    ks.upsert(
        MemoryItem(
            id="acme",
            text="Acme Corp cloud revenue grew in Q3",
            doc_type=DocType.KNOWLEDGE.value,
            score=0.9,
        )
    )
    ks.upsert(
        MemoryItem(
            id="jup",
            text="Ganymede and Callisto are among the largest moons of Jupiter",
            doc_type=DocType.KNOWLEDGE.value,
            score=0.8,
        )
    )
    result = probe_knowledge(ks, "largest moons of Jupiter", max_items=3)
    assert result.hit_count >= 1
    assert all("Acme" not in (h.text or "") for h in result.hits)
    assert result.strength in {"strong", "medium", "weak"}


def _two_endpoint_store() -> EndpointStoreData:
    return EndpointStoreData(
        endpoints=[
            UserEndpoint(
                slot=1,
                label="reasoner",
                model_id="big-reasoner",
                api_base="http://r",
                api_key="k1",
                intended_task="reasoning",
            ),
            UserEndpoint(
                slot=2,
                label="extractor",
                model_id="fast-extract",
                api_base="http://e",
                api_key="k2",
                intended_task="extraction",
            ),
            UserEndpoint(slot=3, label="", model_id="", api_base="", enabled=False),
        ],
        capabilities={
            "big-reasoner": {
                "planner": 0.9,
                "searcher": 0.4,
                "recommender": 0.8,
                "latency_sec": 2.0,
            },
            "fast-extract": {
                "planner": 0.3,
                "searcher": 0.95,
                "recommender": 0.4,
                "latency_sec": 0.4,
            },
        },
    )


def test_policy_extraction_picks_searcher_extract(monkeypatch):
    monkeypatch.setenv("RCR_TASK_MODEL_ROUTING", "true")
    store = _two_endpoint_store()
    registry = ModelRegistry(profile_name="single-default")
    probe = KnowledgeProbeResult(hit_count=1, lexical_coverage=0.2, strength="medium")
    plan = build_run_model_plan(
        query_task="extraction",
        probe=probe,
        registry=registry,
        store=store,
        policy_doc=load_policy(),
        enabled=True,
    )
    assert plan.enabled
    searcher = plan.choices["searcher"]
    assert "extract" in searcher.endpoint.model or searcher.intended_task == "extraction"


def test_policy_weak_escalates_planner(monkeypatch):
    monkeypatch.setenv("RCR_TASK_MODEL_ROUTING", "true")
    store = _two_endpoint_store()
    registry = ModelRegistry(profile_name="single-default")
    probe = KnowledgeProbeResult(hit_count=0, lexical_coverage=0.0, strength="weak")
    plan = build_run_model_plan(
        query_task="reasoning",
        probe=probe,
        registry=registry,
        store=store,
        enabled=True,
    )
    assert plan.enabled
    assert plan.choices["planner"].endpoint.model.endswith("big-reasoner") or (
        "reasoner" in plan.choices["planner"].endpoint.model
    )
    assert "escalate" in plan.choices["planner"].reason


def test_policy_disabled_noop(monkeypatch):
    monkeypatch.delenv("RCR_TASK_MODEL_ROUTING", raising=False)
    store = _two_endpoint_store()
    registry = ModelRegistry(profile_name="single-default")
    plan = build_run_model_plan(
        query_task="extraction",
        probe=KnowledgeProbeResult(strength="strong", hit_count=3, lexical_coverage=0.5),
        registry=registry,
        store=store,
        enabled=False,
    )
    assert plan.enabled is False
    assert plan.note == "routing_disabled"


def test_orchestrator_records_model_plan_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("RCR_USE_OPENSEARCH", "false")
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "true")
    monkeypatch.setenv("RCR_TASK_MODEL_ROUTING", "true")
    monkeypatch.setenv("RCR_USE_USER_ENDPOINTS", "false")
    store_path = tmp_path / "eps.json"
    monkeypatch.setenv("RCR_USER_ENDPOINTS_PATH", str(store_path))

    from rcr_router.endpoint_store import save_store

    store = _two_endpoint_store()
    save_store(store, store_path)

    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    orch.knowledge.upsert(
        MemoryItem(
            id="know-jup",
            text="Ganymede Callisto Io Europa are the largest moons of Jupiter",
            doc_type=DocType.KNOWLEDGE.value,
        )
    )
    result = orch.run(
        "largest moons of Jupiter",
        strategy=RoutingStrategy.RCR,
        max_rounds=1,
        knowledge_writeback=False,
        task_model_routing=True,
    )
    assert result.model_plan.get("enabled") is True
    assert result.model_plan.get("roles")
    assert result.knowledge_probe.get("hit_count", 0) >= 1


def test_orchestrator_skips_plan_when_off(monkeypatch):
    monkeypatch.setenv("RCR_USE_OPENSEARCH", "false")
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "true")
    monkeypatch.delenv("RCR_TASK_MODEL_ROUTING", raising=False)
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    result = orch.run(
        "What drove Acme Corp revenue growth in Q3?",
        strategy=RoutingStrategy.RCR,
        max_rounds=1,
        knowledge_writeback=False,
        task_model_routing=False,
    )
    assert result.model_plan.get("enabled") is False
