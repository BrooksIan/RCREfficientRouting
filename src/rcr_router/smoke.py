"""Smoke run: RCR at T=1 and T=3 (uses mock LLM unless keys are set)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rcr_router.factory import build_orchestrator  # noqa: E402
from rcr_router.llm import build_llm_client  # noqa: E402
from rcr_router.models import DocType, MemoryItem, RoutingStrategy  # noqa: E402


def _seed_knowledge(orch) -> None:
    corpus_path = ROOT / "assets" / "data" / "knowledge-corpus.json"
    rows = json.loads(corpus_path.read_text(encoding="utf-8"))
    texts = [r["text"] for r in rows]
    embs = orch.llm.embed(texts, input_type="passage")
    for row, emb in zip(rows, embs):
        orch.knowledge.upsert(
            MemoryItem(
                id=row["id"],
                text=row["text"],
                doc_type=row.get("doc_type", DocType.KNOWLEDGE.value),
                embedding=emb,
            )
        )


def main() -> None:
    use_os = os.getenv("RCR_USE_OPENSEARCH", "false").lower() in {"1", "true", "yes"}
    # Default to mock unless Cloudera/LiteLLM is configured or user forces real LLM
    if "RCR_USE_MOCK_LLM" not in os.environ:
        from rcr_router.cloudera import cloudera_configured

        os.environ["RCR_USE_MOCK_LLM"] = "false" if cloudera_configured() else "true"
    force_mock = os.getenv("RCR_USE_MOCK_LLM", "true").lower() in {"1", "true", "yes"}
    orch = build_orchestrator(use_opensearch=use_os, force_mock_llm=force_mock)
    _seed_knowledge(orch)

    query = "What drove Acme Corp revenue growth in Q3?"
    for t in (1, 3):
        result = orch.run(query, strategy=RoutingStrategy.RCR, max_rounds=t)
        print(
            f"T={t} strategy={result.strategy} tokens={result.usage['total_tokens']} "
            f"answer={result.answer[:80]!r}"
        )


if __name__ == "__main__":
    main()
