from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .models import MemoryItem
from .store import KnowledgeStore, MemoryStore

WORKING_INDEX = os.getenv("OPENSEARCH_WORKING_INDEX", "rcr-working-memory")
KNOWLEDGE_INDEX = os.getenv("OPENSEARCH_KNOWLEDGE_INDEX", "rcr-knowledge")
DEFAULT_EMBED_DIM = int(os.getenv("OPENSEARCH_EMBED_DIM", "8"))


def _index_refresh() -> bool | str:
    """Per-write refresh is expensive; default off. Set OPENSEARCH_REFRESH=true to force."""
    flag = os.getenv("OPENSEARCH_REFRESH", "false").lower()
    if flag in {"1", "true", "yes"}:
        return True
    return False


def opensearch_client_from_env():
    from opensearchpy import OpenSearch

    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    user = os.getenv("OPENSEARCH_USER") or None
    password = os.getenv("OPENSEARCH_PASSWORD") or None
    use_ssl = os.getenv("OPENSEARCH_USE_SSL", "false").lower() in {"1", "true", "yes"}
    verify = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() in {"1", "true", "yes"}

    http_auth = (user, password) if user and password else None
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=http_auth,
        use_ssl=use_ssl,
        verify_certs=verify,
        ssl_show_warn=False,
    )


def opensearch_available(client=None) -> bool:
    try:
        client = client or opensearch_client_from_env()
        return bool(client.ping())
    except Exception:
        return False


def load_index_template(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        root = Path(__file__).resolve().parents[2]
        path = root / "deploy" / "opensearch_index_templates.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_indices(
    client,
    embed_dim: int = DEFAULT_EMBED_DIM,
    *,
    recreate_on_mismatch: bool = False,
) -> dict[str, str]:
    """Create working/knowledge indices. Optionally drop+recreate on knn dim mismatch."""
    template = load_index_template()
    props = template["template"]["mappings"]["properties"]
    props["embedding"]["dimension"] = embed_dim

    body = {
        "settings": template["template"]["settings"],
        "mappings": template["template"]["mappings"],
    }
    actions: dict[str, str] = {}
    for index in (WORKING_INDEX, KNOWLEDGE_INDEX):
        exists = client.indices.exists(index=index)
        if not exists:
            client.indices.create(index=index, body=body)
            actions[index] = "created"
            continue
        # Detect knn dim mismatch
        try:
            mapping = client.indices.get_mapping(index=index)
            cur = (
                mapping[index]["mappings"]
                .get("properties", {})
                .get("embedding", {})
                .get("dimension")
            )
        except Exception:
            cur = None
        if cur is not None and int(cur) != int(embed_dim):
            if recreate_on_mismatch:
                client.indices.delete(index=index)
                client.indices.create(index=index, body=body)
                actions[index] = f"recreated({cur}->{embed_dim})"
            else:
                actions[index] = f"mismatch({cur}!={embed_dim})"
        else:
            actions[index] = "exists"
    return actions


def delete_session(client, session_id: str, index: str = WORKING_INDEX) -> int:
    """Remove working-memory docs for a session. Returns deleted count."""
    resp = client.delete_by_query(
        index=index,
        body={"query": {"term": {"session_id": session_id}}},
        refresh=True,
    )
    return int(resp.get("deleted", 0))


def clear_working_memory(client=None, index: str = WORKING_INDEX) -> int:
    """Delete all working-memory docs (keeps knowledge corpus). Returns deleted count."""
    client = client or opensearch_client_from_env()
    try:
        if not client.indices.exists(index=index):
            return 0
        resp = client.delete_by_query(
            index=index,
            body={"query": {"match_all": {}}},
            refresh=True,
            conflicts="proceed",
        )
        return int(resp.get("deleted", 0))
    except Exception:
        return 0


def working_memory_count(client=None, index: str = WORKING_INDEX) -> int:
    client = client or opensearch_client_from_env()
    try:
        if not client.indices.exists(index=index):
            return 0
        return int(client.count(index=index)["count"])
    except Exception:
        return 0


def _source_to_item(doc_id: str, src: dict[str, Any], score: float = 0.0) -> MemoryItem:
    payload = dict(src)
    payload["id"] = doc_id
    payload["score"] = score
    return MemoryItem.from_dict(payload)


def _item_body(item: MemoryItem) -> dict[str, Any]:
    body = item.to_dict()
    body.pop("id", None)
    body.pop("score", None)
    if not body.get("embedding"):
        body.pop("embedding", None)
    if not body.get("task"):
        body.pop("task", None)
    if not body.get("struct_key"):
        body.pop("struct_key", None)
    if not body.get("struct_kind"):
        body.pop("struct_kind", None)
    if not body.get("structure"):
        body.pop("structure", None)
    meta = dict(body.get("metadata") or {})
    if item.task and "task" not in meta:
        meta["task"] = item.task
    if item.struct_key:
        meta.setdefault("struct_key", item.struct_key)
    if item.struct_kind:
        meta.setdefault("struct_kind", item.struct_kind)
    if meta:
        body["metadata"] = meta
    else:
        body.pop("metadata", None)
    return body


def _hybrid_bool(
    query: str,
    *,
    filters: list[dict[str, Any]] | None = None,
    embedding: Optional[list[float]] = None,
    k: int = 20,
) -> dict[str, Any]:
    """BM25 match OR k-NN (should), with optional filters."""
    should: list[dict[str, Any]] = [{"match": {"text": {"query": query}}}]
    if embedding:
        should.append({"knn": {"embedding": {"vector": embedding, "k": k}}})
    bool_q: dict[str, Any] = {
        "should": should,
        "minimum_should_match": 1,
    }
    if filters:
        bool_q["filter"] = filters
    return {"bool": bool_q}


class OpenSearchMemoryStore(MemoryStore):
    def __init__(self, client=None, index: str = WORKING_INDEX) -> None:
        self.client = client or opensearch_client_from_env()
        self.index = index

    def upsert(self, item: MemoryItem) -> None:
        self.client.index(
            index=self.index, id=item.id, body=_item_body(item), refresh=_index_refresh()
        )

    def bulk_upsert(self, items: list[MemoryItem]) -> None:
        for item in items:
            self.upsert(item)

    def list_session(self, session_id: str) -> list[MemoryItem]:
        resp = self.client.search(
            index=self.index,
            body={
                "size": 1000,
                "query": {"term": {"session_id": session_id}},
            },
        )
        return [
            _source_to_item(h["_id"], h["_source"], h.get("_score") or 0.0)
            for h in resp["hits"]["hits"]
        ]

    def delete_session(self, session_id: str) -> int:
        return delete_session(self.client, session_id, index=self.index)

    def delete(self, item_id: str) -> None:
        try:
            self.client.delete(
                index=self.index, id=item_id, refresh=_index_refresh() or True
            )
        except Exception:
            pass

    def search(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        role_tag: Optional[str] = None,
        stage_tag: Optional[str] = None,
        task: Optional[str] = None,
        k: int = 20,
        embedding: Optional[list[float]] = None,
    ) -> list[MemoryItem]:
        filters: list[dict[str, Any]] = []
        if session_id is not None:
            filters.append({"term": {"session_id": session_id}})
        if role_tag is not None:
            filters.append({"term": {"role_tag": role_tag}})
        if stage_tag is not None:
            filters.append({"term": {"stage_tag": stage_tag}})
        # Soft task preference via should-boost (not a hard filter)
        should_extra: list[dict[str, Any]] = []
        if task and task != "generic":
            should_extra.append({"term": {"task": {"value": task, "boost": 2.0}}})
            should_extra.append({"term": {"task": {"value": "generic", "boost": 0.5}}})

        query_body = _hybrid_bool(query, filters=filters or None, embedding=embedding, k=k)
        if should_extra:
            query_body["bool"].setdefault("should", []).extend(should_extra)

        resp = self.client.search(
            index=self.index,
            body={
                "size": k,
                "query": query_body,
            },
        )
        return [
            _source_to_item(h["_id"], h["_source"], h.get("_score") or 0.0)
            for h in resp["hits"]["hits"]
        ]


class OpenSearchKnowledgeStore(KnowledgeStore):
    def __init__(self, client=None, index: str = KNOWLEDGE_INDEX) -> None:
        self.client = client or opensearch_client_from_env()
        self.index = index

    def upsert(self, item: MemoryItem) -> None:
        self.client.index(
            index=self.index, id=item.id, body=_item_body(item), refresh=_index_refresh()
        )

    def bulk_upsert(self, items: list[MemoryItem]) -> None:
        for item in items:
            self.upsert(item)

    def delete(self, item_id: str) -> None:
        try:
            self.client.delete(
                index=self.index, id=item_id, refresh=_index_refresh() or True
            )
        except Exception:
            pass

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        embedding: Optional[list[float]] = None,
        task: Optional[str] = None,
        soft_task: bool = True,
    ) -> list[MemoryItem]:
        filters: list[dict[str, Any]] | None = None
        should_extra: list[dict[str, Any]] = []
        if task and task != "generic":
            if soft_task:
                should_extra.append({"term": {"task": {"value": task, "boost": 2.5}}})
                should_extra.append({"term": {"task": {"value": "generic", "boost": 0.6}}})
            else:
                filters = [
                    {
                        "bool": {
                            "should": [
                                {"term": {"task": task}},
                                {"term": {"task": "generic"}},
                                {"bool": {"must_not": {"exists": {"field": "task"}}}},
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                ]

        query_body = _hybrid_bool(query, filters=filters, embedding=embedding, k=k)
        if should_extra:
            query_body["bool"].setdefault("should", []).extend(should_extra)

        resp = self.client.search(
            index=self.index,
            body={
                "size": k,
                "query": query_body,
            },
        )
        return [
            _source_to_item(h["_id"], h["_source"], h.get("_score") or 0.0)
            for h in resp["hits"]["hits"]
        ]
