"""Seed OpenSearch (or in-memory) with roles + knowledge corpus.

CML Job entrypoint referenced from .project-metadata.yaml.

P4: probes live embeddings, syncs ``OPENSEARCH_EMBED_DIM``, and optionally
recreates knn indices when the vector dimension changes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcr_router.embeddings import prepare_embeddings  # noqa: E402
from rcr_router.factory import build_stores  # noqa: E402
from rcr_router.llm import build_llm_client  # noqa: E402
from rcr_router.models import DocType, MemoryItem  # noqa: E402


def main() -> None:
    assets = ROOT / "assets" / "data"
    roles_path = assets / "roles.json"
    corpus_path = assets / "knowledge-corpus.json"

    use_os = os.getenv("RCR_USE_OPENSEARCH", "false").lower() in {"1", "true", "yes"}
    recreate = os.getenv("RCR_RECREATE_INDICES_ON_DIM_MISMATCH", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    llm = build_llm_client()
    status = prepare_embeddings(llm, recreate_on_mismatch=recreate if use_os else False)
    print(
        f"Embeddings: backend={status.backend} dim={status.dim} "
        f"model={status.model or 'n/a'} ok={status.ok} ({status.detail or 'ok'})"
    )
    if not status.ok:
        raise SystemExit(f"Embedding probe failed: {status.detail}")

    memory, knowledge = build_stores(
        use_opensearch=use_os, llm=llm, recreate_on_mismatch=recreate if use_os else False
    )

    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    (assets / "roles.seeded.json").write_text(json.dumps(roles, indent=2), encoding="utf-8")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    texts = [row["text"] for row in corpus]
    embeddings = llm.embed(texts, input_type="passage")
    if embeddings and embeddings[0]:
        dim = len(embeddings[0])
        if any(len(e) != dim for e in embeddings):
            raise SystemExit("Inconsistent embedding dimensions across corpus batch")
        os.environ["OPENSEARCH_EMBED_DIM"] = str(dim)

    items = []
    for row, emb in zip(corpus, embeddings):
        items.append(
            MemoryItem(
                id=row["id"],
                text=row["text"],
                doc_type=row.get("doc_type", DocType.KNOWLEDGE.value),
                role_tag="searcher",
                stage_tag="searching",
                source_agent="seed",
                embedding=emb,
                metadata={
                    "kind": "seed",
                    "embed_backend": getattr(llm, "last_embed_backend", ""),
                    "embed_dim": len(emb),
                },
            )
        )
    knowledge.bulk_upsert(items)

    # Keep a local copy for offline demos (embeddings omitted — large / backend-specific)
    out = assets / "knowledge-corpus.seeded.json"
    out.write_text(
        json.dumps([i.to_dict() | {"embedding": None} for i in items], indent=2),
        encoding="utf-8",
    )

    backend = "opensearch" if use_os else "in-memory"
    print(
        f"Seeded {len(items)} knowledge docs via {backend}; roles={len(roles)}; "
        f"embed_backend={getattr(llm, 'last_embed_backend', '?')} "
        f"dim={os.getenv('OPENSEARCH_EMBED_DIM')}"
    )
    _ = memory


if __name__ == "__main__":
    main()
