"""CML model entrypoint for Role-Based Agent Routing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcr_router.factory import build_orchestrator  # noqa: E402
from rcr_router.models import DocType, MemoryItem, RoutingStrategy  # noqa: E402
import json


def _ensure_corpus(orch) -> None:
    corpus_path = ROOT / "assets" / "data" / "knowledge-corpus.json"
    if not corpus_path.exists():
        return
    # Skip if knowledge already has docs (best-effort for in-memory always reseeds)
    rows = json.loads(corpus_path.read_text(encoding="utf-8"))
    embs = orch.llm.embed([r["text"] for r in rows])
    for row, emb in zip(rows, embs):
        orch.knowledge.upsert(
            MemoryItem(
                id=row["id"],
                text=row["text"],
                doc_type=row.get("doc_type", DocType.KNOWLEDGE.value),
                embedding=emb,
            )
        )


def run_query(args: dict) -> dict:
    prompt = (args or {}).get("prompt") or (args or {}).get("input") or ""
    strategy = (args or {}).get("strategy", os.getenv("RCR_DEFAULT_STRATEGY", "rcr"))
    max_rounds = int((args or {}).get("max_rounds", os.getenv("RCR_MAX_ROUNDS", "3")))
    session_id = (args or {}).get("session_id")

    use_os = os.getenv("RCR_USE_OPENSEARCH", "false").lower() in {"1", "true", "yes"}
    orch = build_orchestrator(use_opensearch=use_os)
    _ensure_corpus(orch)

    if not str(prompt).strip():
        return {"error": "Empty prompt.", "response": None}

    result = orch.run(
        str(prompt),
        strategy=RoutingStrategy(strategy),
        max_rounds=max_rounds,
        session_id=session_id,
    )
    return {
        "response": {
            "answer": result.answer,
            "session_id": result.session_id,
            "strategy": result.strategy,
            "max_rounds": result.max_rounds,
            "usage": result.usage,
            "traces": [
                {
                    "round": t.round_idx,
                    "role": t.role,
                    "model": t.model,
                    "context_tokens": t.context_tokens,
                    "context_texts": t.context_texts,
                    "output": t.output,
                }
                for t in result.traces
            ],
            "model_profile": result.model_profile,
            "memory_snapshot": result.memory_snapshot,
            "elapsed_sec": result.elapsed_sec,
        }
    }


try:
    import cml.models_v1 as models
    import cml.metrics_v1 as metrics  # noqa: F401

    @models.cml_model(metrics=True)
    def api_wrapper(args):
        return run_query(args if isinstance(args, dict) else {"prompt": str(args)})

except ImportError:
    # Local / non-CML usage
    def api_wrapper(args):
        return run_query(args if isinstance(args, dict) else {"prompt": str(args)})


if __name__ == "__main__":
    sample = run_query(
        {"prompt": "What drove Acme Corp revenue growth in Q3?", "strategy": "rcr", "max_rounds": 1}
    )
    print(json.dumps(sample, indent=2)[:2000])
