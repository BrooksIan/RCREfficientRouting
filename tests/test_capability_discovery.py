from __future__ import annotations

from pathlib import Path

from rcr_router.capability_probe import (
    assign_roles,
    prior_from_model_id,
    probe_endpoint,
    run_discovery,
)
from rcr_router.endpoint_store import EndpointStoreData, UserEndpoint, load_store, save_store
from rcr_router.llm import MockLLMClient
from rcr_router.model_routing import ModelRegistry


def test_prior_from_model_id_nemotron():
    scores = prior_from_model_id("nvidia/nemotron-3-super-120b-a12b")
    assert scores["planner"] > scores["router"]


def test_intended_task_boosts_prior():
    from rcr_router.capability_probe import combined_prior, prior_from_intended_task

    coding = prior_from_intended_task("coding")
    assert coding["planner"] > coding["router"]
    extraction = prior_from_intended_task("extraction")
    assert extraction["searcher"] > extraction["planner"]
    blended = combined_prior("vendor/generic-model", "routing")
    plain = combined_prior("vendor/generic-model", "generic")
    assert blended["router"] > plain["router"]


def test_probe_and_assign_with_mock(tmp_path: Path, monkeypatch):
    store_path = tmp_path / "endpoints.json"
    monkeypatch.setenv("RCR_USER_ENDPOINTS_PATH", str(store_path))
    store = EndpointStoreData(
        endpoints=[
            UserEndpoint(
                slot=1,
                label="Reasoner",
                model_id="nvidia/nemotron-3-super-120b-a12b",
                api_base="https://example.com/v1",
                api_key="k1",
                intended_task="reasoning",
            ),
            UserEndpoint(
                slot=2,
                label="Extractor",
                model_id="vendor/mini-extract",
                api_base="https://example.com/v1",
                api_key="k2",
                intended_task="extraction",
            ),
            UserEndpoint(
                slot=3,
                label="General",
                model_id="vendor/gpt-4-large",
                api_base="https://example.com/v1",
                api_key="k3",
                intended_task="synthesis",
            ),
        ]
    )
    # Probe each with mock client
    for ep in store.endpoints:
        scores = probe_endpoint(ep, llm=MockLLMClient())
        store.capabilities[ep.model_id] = scores.as_dict()
        store.capabilities[str(ep.slot)] = scores.as_dict()
    store.role_assignments = assign_roles(store)
    save_store(store, store_path)

    assert set(store.role_assignments) >= {"planner", "searcher", "recommender"}
    assert all("intended_task" in info for info in store.role_assignments.values())
    # Extraction-tagged model should win searcher when probes are otherwise equal.
    assert store.role_assignments["searcher"]["intended_task"] == "extraction"
    # Prefer unique slots when possible
    slots = [info["slot"] for info in store.role_assignments.values()]
    assert len(slots) == len(set(slots)) or len(store.endpoints) < 4


def test_load_store_defaults_intended_task(tmp_path: Path, monkeypatch):
    store_path = tmp_path / "legacy.json"
    monkeypatch.setenv("RCR_USER_ENDPOINTS_PATH", str(store_path))
    store_path.write_text(
        '{"endpoints":[{"slot":1,"label":"A","model_id":"m","api_base":"https://x/v1","api_key":"k"}],'
        '"capabilities":{},"role_assignments":{},"last_probed_at":""}',
        encoding="utf-8",
    )
    loaded = load_store(store_path)
    assert loaded.endpoints[0].normalized_task() == "generic"


def test_run_discovery_mock(tmp_path: Path, monkeypatch):
    store_path = tmp_path / "ep.json"
    monkeypatch.setenv("RCR_USER_ENDPOINTS_PATH", str(store_path))
    store = EndpointStoreData(
        endpoints=[
            UserEndpoint(
                slot=1,
                label="A",
                model_id="nvidia/nemotron-3-super-120b-a12b",
                api_base="https://a/v1",
                api_key="x",
            ),
            UserEndpoint(
                slot=2,
                label="B",
                model_id="vendor/mini-flash",
                api_base="https://b/v1",
                api_key="y",
            ),
            UserEndpoint(slot=3, label="Empty", model_id="", api_base="", enabled=False),
        ]
    )
    save_store(store, store_path)
    out = run_discovery(load_store(store_path), use_mock=True)
    assert out.last_probed_at
    assert out.role_assignments
    assert "planner" in out.role_assignments


def test_registry_uses_discovered_assignments(tmp_path: Path, monkeypatch):
    store_path = tmp_path / "disc.json"
    monkeypatch.setenv("RCR_USER_ENDPOINTS_PATH", str(store_path))
    monkeypatch.setenv("RCR_USE_USER_ENDPOINTS", "true")
    monkeypatch.delenv("RCR_FORCE_DIRECT_CAIIS", raising=False)
    monkeypatch.delenv("CLOUDERA_AI_INFERENCE_API_BASE", raising=False)
    store = EndpointStoreData(
        endpoints=[
            UserEndpoint(
                slot=1,
                label="P",
                model_id="plan-model",
                api_base="https://plan/v1",
                api_key="a",
            ),
            UserEndpoint(
                slot=2,
                label="S",
                model_id="search-model",
                api_base="https://search/v1",
                api_key="b",
            ),
            UserEndpoint(
                slot=3,
                label="R",
                model_id="rec-model",
                api_base="https://rec/v1",
                api_key="c",
            ),
        ],
        role_assignments={
            "planner": {
                "slot": 1,
                "label": "P",
                "model_id": "plan-model",
                "api_base": "https://plan/v1",
                "api_key": "a",
                "score": 0.9,
            },
            "searcher": {
                "slot": 2,
                "label": "S",
                "model_id": "search-model",
                "api_base": "https://search/v1",
                "api_key": "b",
                "score": 0.8,
            },
            "recommender": {
                "slot": 3,
                "label": "R",
                "model_id": "rec-model",
                "api_base": "https://rec/v1",
                "api_key": "c",
                "score": 0.85,
            },
        },
    )
    save_store(store, store_path)
    reg = ModelRegistry(profile_name="single-default")
    assert reg.for_role("planner").model == "openai/plan-model"
    assert reg.for_role("searcher").api_base.endswith("/search/v1")
    assert reg.profile_name == "user-discovered"


def test_org_scoped_model_gets_openai_prefix(tmp_path: Path, monkeypatch):
    from rcr_router.model_routing import as_openai_compatible_model

    assert as_openai_compatible_model("defog/llama-3-sqlcoder-8b") == (
        "openai/defog/llama-3-sqlcoder-8b"
    )
    assert as_openai_compatible_model("openai/nvidia/nemotron") == "openai/nvidia/nemotron"
    store_path = tmp_path / "org.json"
    monkeypatch.setenv("RCR_USER_ENDPOINTS_PATH", str(store_path))
    monkeypatch.setenv("RCR_USE_USER_ENDPOINTS", "true")
    monkeypatch.delenv("CLOUDERA_AI_INFERENCE_API_BASE", raising=False)
    store = EndpointStoreData(
        endpoints=[
            UserEndpoint(
                slot=1,
                label="SQL",
                model_id="defog/llama-3-sqlcoder-8b",
                api_base="https://example/v1",
                api_key="tok",
            ),
            UserEndpoint(slot=2, label="", model_id="", api_base="", enabled=False),
            UserEndpoint(slot=3, label="", model_id="", api_base="", enabled=False),
        ],
        role_assignments={
            "planner": {
                "slot": 1,
                "label": "SQL",
                "model_id": "defog/llama-3-sqlcoder-8b",
                "api_base": "https://example/v1",
                "api_key": "tok",
                "score": 0.9,
            }
        },
    )
    save_store(store, store_path)
    reg = ModelRegistry(profile_name="single-default")
    assert reg.for_role("planner").model == "openai/defog/llama-3-sqlcoder-8b"
