from __future__ import annotations

import os

from .llm import LLMClient, MockLLMClient, build_llm_client
from .memory_store_memory import InMemoryKnowledgeStore, InMemoryMemoryStore
from .model_routing import ModelRegistry, active_model_registry
from .orchestrator import Orchestrator
from .store import KnowledgeStore, MemoryStore


def build_stores(
    *,
    use_opensearch: bool | None = None,
    llm: LLMClient | None = None,
    recreate_on_mismatch: bool | None = None,
) -> tuple[MemoryStore, KnowledgeStore]:
    if use_opensearch is None:
        flag = os.getenv("RCR_USE_OPENSEARCH", "").strip().lower()
        if flag in {"1", "true", "yes"}:
            use_opensearch = True
        elif flag in {"0", "false", "no"}:
            use_opensearch = False
        else:
            # Auto-enable when OpenSearch host/port is configured.
            use_opensearch = bool(
                os.getenv("OPENSEARCH_HOST") or os.getenv("OPENSEARCH_PORT")
            )

    if not use_opensearch:
        return InMemoryMemoryStore(), InMemoryKnowledgeStore()

    from .embeddings import prepare_embeddings
    from .opensearch_store import (
        OpenSearchKnowledgeStore,
        OpenSearchMemoryStore,
        opensearch_client_from_env,
    )

    client = opensearch_client_from_env()
    llm = llm or build_llm_client()
    status = prepare_embeddings(llm, recreate_on_mismatch=recreate_on_mismatch)
    dim = status.dim if status.ok and status.dim > 0 else int(
        os.getenv("OPENSEARCH_EMBED_DIM", "8")
    )
    os.environ["OPENSEARCH_EMBED_DIM"] = str(dim)
    return OpenSearchMemoryStore(client), OpenSearchKnowledgeStore(client)


def build_orchestrator(
    *,
    use_opensearch: bool | None = None,
    force_mock_llm: bool | None = None,
    registry: ModelRegistry | None = None,
) -> Orchestrator:
    llm = build_llm_client(force_mock=force_mock_llm)
    memory, knowledge = build_stores(use_opensearch=use_opensearch, llm=llm)
    reg = registry or active_model_registry()
    return Orchestrator(memory, knowledge, llm=llm, registry=reg)
