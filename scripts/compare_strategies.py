"""Offline comparison harness: Full vs Static vs RCR on the fixture set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcr_router.factory import build_orchestrator
from rcr_router.metrics import summarize_run
from rcr_router.models import DocType, MemoryItem, RoutingStrategy


FIXTURE_QUERIES = [
    "What drove Acme Corp revenue growth in Q3?",
    "How did Acme Corp cloud margin change in Q3?",
    "Was Acme Corp Q2 growth driven by cloud or hardware?",
]


def seed(orch) -> None:
    corpus = json.loads((ROOT / "assets" / "data" / "knowledge-corpus.json").read_text())
    embs = orch.llm.embed([r["text"] for r in corpus])
    for row, emb in zip(corpus, embs):
        orch.knowledge.upsert(
            MemoryItem(
                id=row["id"],
                text=row["text"],
                doc_type=row.get("doc_type", DocType.KNOWLEDGE.value),
                embedding=emb,
            )
        )


def compare(max_rounds: int = 2, *, use_judge: bool = False) -> list[dict]:
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    seed(orch)
    rows: list[dict] = []
    for query in FIXTURE_QUERIES:
        for strategy in (RoutingStrategy.FULL, RoutingStrategy.STATIC, RoutingStrategy.RCR):
            result = orch.run(
                query,
                strategy=strategy,
                max_rounds=max_rounds,
                knowledge_writeback=False,
            )
            metrics = summarize_run(
                result, query, llm=orch.llm if use_judge else None, use_judge=use_judge
            ).to_dict()
            metrics["query"] = query
            metrics["answer_preview"] = result.answer[:100]
            rows.append(metrics)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Full/Static/RCR routing")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also run paper LLM Answer Quality Score (mock offline)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "assets" / "data" / "eval-comparison.json",
    )
    args = parser.parse_args()
    rows = compare(max_rounds=args.max_rounds, use_judge=args.judge)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    hdr = f"{'strategy':8} {'prompt':>8} {'compl':>8} {'ctx':>8} {'lex':>7}"
    if args.judge:
        hdr += f" {'judge':>7}"
    hdr += " query"
    print(hdr)
    for r in rows:
        line = (
            f"{r['strategy']:8} {r['prompt_tokens']:8} {r['completion_tokens']:8} "
            f"{r['context_tokens']:8} {r['answer_quality']:7.2f}"
        )
        if args.judge:
            jq = r.get("judge_quality")
            line += f" {jq:7.2f}" if jq is not None else f" {'n/a':>7}"
        line += f" {r['query'][:44]}"
        print(line)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
