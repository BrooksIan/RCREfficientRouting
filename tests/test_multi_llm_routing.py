from __future__ import annotations

from rcr_router.agent_router import AgentActivationRouter
from rcr_router.llm import MockLLMClient
from rcr_router.model_routing import ModelRegistry
from rcr_router.models import AgentRole


def test_single_profile_inherits_default(monkeypatch):
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("RCR_FORCE_DIRECT_CAIIS", raising=False)
    monkeypatch.setenv("RCR_FORCE_DIRECT_CAIIS", "true")
    monkeypatch.setenv(
        "CLOUDERA_AI_INFERENCE_API_BASE",
        "https://example.com/namespaces/serving-default/endpoints/demo/v1",
    )
    monkeypatch.setenv("CLOUDERA_AI_INFERENCE_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("CDP_TOKEN", "token")
    monkeypatch.setenv("RCR_MODEL_PROFILE", "single-nemotron")
    reg = ModelRegistry()
    assert reg.for_role("planner").model == "openai/nvidia/nemotron-3-super-120b-a12b"
    assert reg.for_role("searcher").model == reg.for_role("recommender").model


def test_env_override_per_role(monkeypatch):
    monkeypatch.setenv("RCR_FORCE_DIRECT_CAIIS", "true")
    monkeypatch.setenv(
        "CLOUDERA_AI_INFERENCE_API_BASE",
        "https://example.com/namespaces/serving-default/endpoints/demo/v1",
    )
    monkeypatch.setenv("CLOUDERA_AI_INFERENCE_MODEL", "shared-model")
    monkeypatch.setenv("CDP_TOKEN", "token")
    monkeypatch.setenv("RCR_MODEL_PROFILE", "split-direct")
    monkeypatch.setenv("LITELLM_MODEL_SEARCHER", "small-search-model")
    monkeypatch.setenv(
        "CLOUDERA_AI_INFERENCE_API_BASE_SEARCHER",
        "https://example.com/namespaces/serving-default/endpoints/searcher/v1",
    )
    reg = ModelRegistry()
    assert "small-search-model" in reg.for_role("searcher").model
    assert reg.for_role("searcher").api_base.endswith("/searcher/v1")
    assert "shared-model" in reg.for_role("planner").model


def test_split_via_proxy_aliases(monkeypatch):
    monkeypatch.delenv("RCR_FORCE_DIRECT_CAIIS", raising=False)
    monkeypatch.delenv("CLOUDERA_AI_INFERENCE_API_BASE", raising=False)
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk")
    monkeypatch.setenv("LITELLM_MODEL", "cloudera-chat")
    monkeypatch.setenv("RCR_MODEL_PROFILE", "split-via-proxy")
    reg = ModelRegistry()
    assert reg.for_role("planner").model == "cloudera-planner"
    assert reg.for_role("recommender").model == "cloudera-recommender"
    assert reg.for_role("router").model == "cloudera-router"


def test_llm_agent_router_enabled(monkeypatch):
    monkeypatch.setenv("RCR_LLM_AGENT_ROUTER", "true")
    llm = MockLLMClient()
    act = AgentActivationRouter(llm, enabled=True)
    roles = act.select("What drove Acme revenue?", round_idx=1)
    assert AgentRole.PLANNER in roles
    assert AgentRole.SEARCHER in roles


def test_orchestrator_records_models(monkeypatch):
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "true")
    monkeypatch.setenv("RCR_MODEL_PROFILE", "single-default")
    monkeypatch.setenv("RCR_LLM_AGENT_ROUTER", "false")
    from rcr_router.factory import build_orchestrator
    from rcr_router.models import DocType, MemoryItem, RoutingStrategy

    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    orch.knowledge.upsert(
        MemoryItem(id="k1", text="Acme Corp cloud revenue grew in Q3", doc_type=DocType.KNOWLEDGE.value)
    )
    result = orch.run("Acme Q3 revenue?", strategy=RoutingStrategy.RCR, max_rounds=1)
    assert result.model_profile.get("profile") == "single-default"
    assert all(t.model for t in result.traces)
