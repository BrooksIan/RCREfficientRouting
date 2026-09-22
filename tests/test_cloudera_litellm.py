from __future__ import annotations

import json
from pathlib import Path

from rcr_router.cloudera import (
    cloudera_configured,
    resolve_api_key,
    resolve_cloudera_settings,
)
from rcr_router.llm import LiteLLMClient, MockLLMClient, build_llm_client, role_model_map


def test_resolve_api_key_prefers_cdp(monkeypatch):
    monkeypatch.setenv("CDP_TOKEN", "cdp-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    assert resolve_api_key() == "cdp-secret"


def test_resolve_api_key_from_jwt_file(monkeypatch, tmp_path: Path):
    jwt_path = tmp_path / "jwt"
    jwt_path.write_text(json.dumps({"access_token": "jwt-token"}), encoding="utf-8")
    monkeypatch.delenv("CDP_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDERA_AI_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CML_JWT_PATH", str(jwt_path))
    assert resolve_api_key() == "jwt-token"


def test_proxy_settings(monkeypatch):
    monkeypatch.delenv("RCR_FORCE_DIRECT_CAIIS", raising=False)
    monkeypatch.delenv("CLOUDERA_AI_INFERENCE_API_BASE", raising=False)
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-rcr-local")
    monkeypatch.setenv("LITELLM_MODEL", "cloudera-chat")
    monkeypatch.setenv("LITELLM_EMBED_MODEL", "cloudera-embed")
    settings = resolve_cloudera_settings()
    assert settings is not None
    assert settings.via_proxy is True
    assert settings.litellm_chat_model == "cloudera-chat"
    assert settings.api_base == "http://localhost:4000"


def test_direct_caiis_settings_prefix_openai(monkeypatch):
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.setenv("RCR_FORCE_DIRECT_CAIIS", "true")
    monkeypatch.setenv(
        "CLOUDERA_AI_INFERENCE_API_BASE",
        "https://example.com/namespaces/serving-default/endpoints/demo/v1",
    )
    monkeypatch.setenv("CLOUDERA_AI_INFERENCE_MODEL", "my-nim-model")
    monkeypatch.setenv("CDP_TOKEN", "token")
    settings = resolve_cloudera_settings()
    assert settings is not None
    assert settings.via_proxy is False
    assert settings.litellm_chat_model == "openai/my-nim-model"


def test_build_llm_client_uses_cloudera(monkeypatch):
    monkeypatch.setenv("RCR_USE_MOCK_LLM", "false")
    monkeypatch.delenv("RCR_FORCE_DIRECT_CAIIS", raising=False)
    monkeypatch.delenv("CLOUDERA_AI_INFERENCE_API_BASE", raising=False)
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-rcr-local")
    monkeypatch.setenv("LITELLM_MODEL", "cloudera-chat")
    client = build_llm_client(force_mock=False)
    assert isinstance(client, LiteLLMClient)
    assert client.api_base == "http://localhost:4000"
    assert client.chat_model == "cloudera-chat"


def test_build_llm_client_mock_when_forced(monkeypatch):
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-rcr-local")
    client = build_llm_client(force_mock=True)
    assert isinstance(client, MockLLMClient)


def test_role_model_map_defaults_to_cloudera_alias(monkeypatch):
    monkeypatch.delenv("RCR_FORCE_DIRECT_CAIIS", raising=False)
    monkeypatch.delenv("CLOUDERA_AI_INFERENCE_API_BASE", raising=False)
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk")
    monkeypatch.setenv("LITELLM_MODEL", "cloudera-chat")
    monkeypatch.setenv("LITELLM_MODEL_RECOMMENDER", "cloudera-chat-alt")
    mapping = role_model_map()
    assert mapping["planner"] == "cloudera-chat"
    assert mapping["recommender"] == "cloudera-chat-alt"


def test_cloudera_configured_false_without_creds(monkeypatch):
    for key in (
        "LITELLM_API_BASE",
        "LITELLM_API_KEY",
        "OPENAI_API_KEY",
        "CDP_TOKEN",
        "CLOUDERA_AI_INFERENCE_API_BASE",
        "CLOUDERA_AI_INFERENCE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CML_JWT_PATH", "/tmp/does-not-exist-rcr-jwt")
    assert cloudera_configured() is False
