"""P4: real vs mock vs local embeddings + OpenSearch dim sync."""

from __future__ import annotations

from rcr_router.embeddings import (
    embed_backend_preference,
    local_semantic_embed,
    looks_like_chat_only_model,
    probe_embeddings,
    remote_embed_configured,
    resolve_embed_backend,
)
from rcr_router.llm import LiteLLMClient, MockLLMClient
from rcr_router.opensearch_store import ensure_indices


def test_nemotron_is_chat_only():
    assert looks_like_chat_only_model("nvidia/nemotron-3-super-120b-a12b")
    assert looks_like_chat_only_model("openai/nvidia/nemotron-3-super-120b-a12b")
    assert not looks_like_chat_only_model("text-embedding-3-small")
    assert not looks_like_chat_only_model("BAAI/bge-small-en-v1.5")
    assert not looks_like_chat_only_model("nvidia/llama-3.2-nv-embedqa-1b-v2")


def test_resolve_backend_mock_forced(monkeypatch):
    monkeypatch.setenv("RCR_EMBED_BACKEND", "mock")
    assert resolve_embed_backend(embed_model="text-embedding-3-small") == "mock"
    assert embed_backend_preference() == "mock"


def test_resolve_backend_auto_uses_local_without_real_embed(monkeypatch):
    monkeypatch.setenv("RCR_EMBED_BACKEND", "auto")
    for k in (
        "CLOUDERA_AI_INFERENCE_EMBED_MODEL",
        "CAIIS_EMBED_MODEL",
        "LITELLM_EMBED_MODEL",
        "CLOUDERA_AI_INFERENCE_EMBED_API_BASE",
        "CAIIS_EMBED_API_BASE",
        "CLOUDERA_AI_INFERENCE_API_BASE",
        "LITELLM_API_BASE",
        "OPENAI_API_BASE",
        "CDP_TOKEN",
        "LITELLM_API_KEY",
        "OPENAI_API_KEY",
        "CLOUDERA_AI_INFERENCE_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    assert remote_embed_configured() is False
    assert resolve_embed_backend() == "local"


def test_resolve_backend_rejects_nemotron_as_remote(monkeypatch):
    monkeypatch.setenv("RCR_EMBED_BACKEND", "auto")
    for k in (
        "CLOUDERA_AI_INFERENCE_EMBED_MODEL",
        "CAIIS_EMBED_MODEL",
        "CLOUDERA_AI_INFERENCE_EMBED_API_BASE",
        "CAIIS_EMBED_API_BASE",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LITELLM_EMBED_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-key")
    assert remote_embed_configured() is False
    assert resolve_embed_backend() == "local"


def test_resolve_backend_auto_with_real_embed(monkeypatch):
    monkeypatch.setenv("RCR_EMBED_BACKEND", "auto")
    monkeypatch.setenv("LITELLM_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-key")
    assert resolve_embed_backend() == "litellm"


def test_local_semantic_dim_and_similarity():
    a, label = local_semantic_embed(["Acme Corp cloud revenue growth in Q3"], dim=384)
    b, _ = local_semantic_embed(["Acme cloud subscription revenue"], dim=384)
    c, _ = local_semantic_embed(["pasta cafeteria seasonal menu"], dim=384)
    assert label.startswith("local:")
    assert len(a[0]) == 384

    def cos(u, v):
        return sum(x * y for x, y in zip(u, v))

    assert cos(a[0], b[0]) > cos(a[0], c[0])


def test_mock_probe():
    llm = MockLLMClient(dim=8)
    status = probe_embeddings(llm)
    assert status.ok
    assert status.backend == "mock"
    assert status.dim == 8


def test_litellm_client_uses_local_for_nemotron(monkeypatch):
    monkeypatch.setenv("RCR_EMBED_BACKEND", "auto")
    monkeypatch.setenv("RCR_LOCAL_EMBED_DIM", "384")
    monkeypatch.setenv("OPENSEARCH_EMBED_DIM", "384")
    for k in (
        "CLOUDERA_AI_INFERENCE_EMBED_MODEL",
        "CAIIS_EMBED_MODEL",
        "LITELLM_EMBED_MODEL",
        "CLOUDERA_AI_INFERENCE_EMBED_API_BASE",
        "CAIIS_EMBED_API_BASE",
    ):
        monkeypatch.delenv(k, raising=False)
    client = LiteLLMClient(
        chat_model="openai/nvidia/nemotron-3-super-120b-a12b",
        embed_model="openai/nvidia/nemotron-3-super-120b-a12b",
        api_base="http://example.invalid/v1",
        api_key="sk-test",
        embed_api_base=None,
    )
    client.embed_api_base = None
    vecs = client.embed(["hello cloud revenue"])
    assert str(client.last_embed_backend).startswith("local")
    assert len(vecs[0]) == 384


def test_litellm_client_honors_mock_backend(monkeypatch):
    monkeypatch.setenv("RCR_EMBED_BACKEND", "mock")
    monkeypatch.setenv("OPENSEARCH_EMBED_DIM", "8")
    client = LiteLLMClient(
        chat_model="openai/x",
        embed_model="openai/y",
        api_base="http://example.invalid/v1",
        api_key="sk-test",
    )
    vecs = client.embed(["hello"])
    assert client.last_embed_backend == "mock"
    assert len(vecs) == 1 and len(vecs[0]) == 8


def test_ensure_indices_returns_actions():
    import inspect

    sig = inspect.signature(ensure_indices)
    assert "recreate_on_mismatch" in sig.parameters
