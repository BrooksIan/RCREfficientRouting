#!/usr/bin/env python3
"""Multi-hop QA experiment: Full vs Static vs RCR on HotPot/MuSiQue/2Wiki-style fixtures.

Mirrors the RCR-Router paper claim (arXiv:2508.04903 §Experiments) on a controlled
AMP corpus — not the official HotPotQA / MuSiQue / 2WikiMultihop dumps.

Working memory is intentionally oversized (gold supports + distractors) so Full-context
routing approaches the paper-scale cap while RCR fills under B_i.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Experiment defaults before factory/env reads.
os.environ.setdefault("RCR_BUDGET_PRESET", "savings")
os.environ.setdefault("RCR_USE_MOCK_LLM", "true")
os.environ.setdefault("RCR_TASK_MODEL_ROUTING", "0")
os.environ.setdefault("RCR_KNOWLEDGE_WRITEBACK", "0")

from rcr_router.factory import build_orchestrator
from rcr_router.llm import LLMResponse, MockLLMClient
from rcr_router.metrics import summarize_run
from rcr_router.models import AgentRole, DocType, MemoryItem, RoutingStrategy, TaskStage
from rcr_router.tokens import count_tokens, token_backend

DATA = ROOT / "assets" / "data" / "experiments" / "multihop"
QUESTIONS_PATH = DATA / "questions.json"
PASSAGES_PATH = DATA / "passages.json"
DEFAULT_RESULTS = DATA / "results.json"
DEFAULT_REPORT = ROOT / "docs" / "experiments" / "multihop-rcr-token-savings.md"

_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "are",
    "that",
    "this",
    "for",
    "with",
    "who",
    "what",
    "which",
    "when",
    "where",
    "was",
    "were",
    "by",
    "from",
}


def _terms(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in _STOP
    }


def token_f1(pred: str, gold: str) -> float:
    """Unigram F1 between prediction and gold answer (SQuAD-style)."""
    p = _terms(pred)
    g = _terms(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = len(p & g)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def support_recall(context_texts: list[str], support_passages: list[dict]) -> float:
    """Fraction of gold supports whose distinctive terms appear in routed context."""
    if not support_passages:
        return 0.0
    blob = " ".join(context_texts).lower()
    hits = 0
    for p in support_passages:
        # Require at least 2 content terms from the passage title+text head
        keys = _terms(f"{p.get('title', '')} {p.get('text', '')}")
        # Prefer rare-ish tokens (len>=5) when available
        keys = {t for t in keys if len(t) >= 5} or keys
        need = list(keys)[:6]
        if not need:
            continue
        matched = sum(1 for t in need if t in blob)
        if matched >= max(2, len(need) // 3):
            hits += 1
    return hits / len(support_passages)


class ExperimentMockLLM(MockLLMClient):
    """Mock chat that returns gold when enough supporting evidence is in context."""

    def __init__(self, gold_by_query: dict[str, str], support_terms: dict[str, set[str]]) -> None:
        super().__init__()
        self.gold_by_query = gold_by_query
        self.support_terms = support_terms

    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> LLMResponse:
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        ul = user.lower()
        sl = system.lower()

        # Reuse base mock for non-answer turns
        if "recommender" not in sl and "synthesis quality" not in sl:
            resp = super().complete(
                messages, model=model, api_base=api_base, api_key=api_key
            )
            # Prefer tiktoken-ish prompt accounting for experiment tables
            resp = LLMResponse(
                text=resp.text,
                prompt_tokens=count_tokens(user),
                completion_tokens=count_tokens(resp.text),
                model=resp.model,
            )
            return resp

        query = ""
        if "user query:" in ul:
            query = user.split("User query:", 1)[-1].split("Routed context:", 1)[0].strip()
        gold = self.gold_by_query.get(query.strip())
        terms = self.support_terms.get(query.strip(), set())
        ctx = ""
        if "routed context:" in ul:
            ctx = user.lower().split("routed context:", 1)[-1]
        covered = sum(1 for t in terms if t in ctx) if terms else 0
        need = max(2, len(terms) // 3) if terms else 99
        if gold and covered >= need:
            text = gold
        elif gold:
            text = (
                "Insufficient supporting evidence in routed context; "
                "cannot complete the multi-hop answer."
            )
        else:
            text = f"ANSWER: based on plan and evidence for: {query[:120]}"
        return LLMResponse(
            text=text,
            prompt_tokens=count_tokens(user),
            completion_tokens=count_tokens(text),
            model=model or "mock-llm-experiment",
        )


def load_pack() -> tuple[list[dict], dict[str, dict], list[dict]]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    passages = json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in passages["supporting"]}
    by_id.update({p["id"]: p for p in passages["distractors"]})
    return questions, by_id, passages["distractors"]


def seed_knowledge(orch, by_id: dict[str, dict]) -> None:
    texts = [row["text"] for row in by_id.values()]
    embs = orch.llm.embed(texts, input_type="passage")
    for (pid, row), emb in zip(by_id.items(), embs):
        orch.knowledge.upsert(
            MemoryItem(
                id=pid,
                text=row["text"],
                doc_type=DocType.KNOWLEDGE.value,
                embedding=emb,
                token_length=count_tokens(row["text"]),
                metadata={"title": row.get("title", ""), "canonical": True},
            )
        )


def prefill_working_memory(
    orch,
    *,
    session_id: str,
    support_ids: list[str],
    by_id: dict[str, dict],
    distractors: list[dict],
    query_task: str = "reasoning",
) -> int:
    """Inflate M_t with gold supports + distractors (simulates accumulated multi-hop memory)."""
    items: list[MemoryItem] = []
    # Gold supports first (slightly newer timestamps via insertion order / ids)
    for i, sid in enumerate(support_ids):
        row = by_id[sid]
        text = f"{row.get('title', sid)}. {row['text']}"
        emb = orch.llm.embed([text], input_type="passage")[0]
        items.append(
            MemoryItem(
                id=f"wm-support-{sid}-{session_id}",
                text=text,
                role_tag=AgentRole.SEARCHER.value,
                stage_tag=TaskStage.SEARCHING.value,
                source_agent="knowledge",
                session_id=session_id,
                round=0,
                doc_type=DocType.KNOWLEDGE.value,
                embedding=emb,
                token_length=count_tokens(text),
                task=query_task,
                metadata={"support": True, "passage_id": sid},
            )
        )
    for i, row in enumerate(distractors):
        text = f"{row.get('title', row['id'])}. {row['text']}"
        emb = orch.llm.embed([text], input_type="passage")[0]
        items.append(
            MemoryItem(
                id=f"wm-dist-{row['id']}-{session_id}",
                text=text,
                role_tag=AgentRole.SEARCHER.value,
                stage_tag=TaskStage.SEARCHING.value,
                source_agent="prior",
                session_id=session_id,
                round=0,
                doc_type=DocType.FACT.value,
                embedding=emb,
                token_length=count_tokens(text),
                task=query_task,
                metadata={"distractor": True, "passage_id": row["id"]},
            )
        )
    orch.memory.bulk_upsert(items)
    return sum(i.token_length or 0 for i in items)


def aggregate(rows: list[dict]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_strategy[r["strategy"]].append(r)

    summary: dict[str, Any] = {"by_strategy": {}, "savings_vs_full": {}}
    full_ctx = None
    full_prompt = None
    full_f1 = None
    for strat, items in by_strategy.items():
        n = len(items)
        avg_ctx = sum(i["context_tokens"] for i in items) / n
        avg_prompt = sum(i["prompt_tokens"] for i in items) / n
        avg_f1 = sum(i["gold_f1"] for i in items) / n
        avg_lex = sum(i["answer_quality"] for i in items) / n
        avg_sup = sum(i["support_recall"] for i in items) / n
        summary["by_strategy"][strat] = {
            "n": n,
            "avg_context_tokens": round(avg_ctx, 1),
            "avg_prompt_tokens": round(avg_prompt, 1),
            "avg_gold_f1": round(avg_f1, 4),
            "avg_lexical_quality": round(avg_lex, 3),
            "avg_support_recall": round(avg_sup, 4),
        }
        if strat == "full":
            full_ctx = avg_ctx
            full_prompt = avg_prompt
            full_f1 = avg_f1

    if full_ctx and full_ctx > 0:
        for strat, stats in summary["by_strategy"].items():
            if strat == "full":
                continue
            ctx_save = (full_ctx - stats["avg_context_tokens"]) / full_ctx
            prompt_save = (
                (full_prompt - stats["avg_prompt_tokens"]) / full_prompt if full_prompt else 0.0
            )
            f1_delta = stats["avg_gold_f1"] - (full_f1 or 0.0)
            summary["savings_vs_full"][strat] = {
                "context_token_reduction_pct": round(100 * ctx_save, 1),
                "prompt_token_reduction_pct": round(100 * prompt_save, 1),
                "gold_f1_delta": round(f1_delta, 4),
            }
    return summary


def by_benchmark(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["benchmark"]].append(r)
    for bench, items in groups.items():
        out[bench] = aggregate(items)
    return out


def write_report(
    path: Path,
    *,
    rows: list[dict],
    summary: dict[str, Any],
    bench: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = summary["by_strategy"]
    sav = summary.get("savings_vs_full", {})
    rcr_save = sav.get("rcr", {})
    lines = [
        "# Multi-hop RCR experiment: token savings vs answer quality",
        "",
        "## Claim under test",
        "",
        "> Experiments on three multi-hop QA benchmarks — HotPotQA, MuSiQue, and "
        "2WikiMultihop — demonstrate that RCR-Router reduces token usage "
        "(up to 30%) while improving or maintaining answer quality.",
        "",
        f"Paper: [arXiv:2508.04903](https://arxiv.org/abs/2508.04903) §Experiments. "
        f"This AMP run is a **controlled replication-style experiment** on synthetic "
        f"fixtures in those benchmark *styles* (not the official full dumps).",
        "",
        "## Methodology",
        "",
        f"- **Questions:** {meta['n_questions']} multi-hop items "
        f"({meta['benchmarks']}) in `assets/data/experiments/multihop/`.",
        f"- **Working memory:** each run prefills M_t with gold supporting passages "
        f"plus {meta['n_distractors']} distractors (~{meta['avg_prefill_tokens']} tokens) "
        f"so Full-context routing can approach `full_context_cap`.",
        f"- **Strategies:** Full / Static / RCR; budget preset `{meta['budget_preset']}`; "
        f"token backend `{meta['token_backend']}`; max_rounds={meta['max_rounds']}.",
        "- **LLM:** deterministic experiment mock that emits the gold answer only when "
        "enough supporting evidence appears in routed context (isolates routing quality).",
        "- **Metrics:** routed `context_tokens`, LLM `prompt_tokens`, gold-answer unigram F1, "
        "supporting-passage recall in routed context, lexical answer quality [1–5].",
        "",
        "## Results (aggregate)",
        "",
        "| Strategy | Avg context tokens | Avg prompt tokens | Gold F1 | Support recall | Lexical Q |",
        "|----------|-------------------:|------------------:|--------:|---------------:|----------:|",
    ]
    for name in ("full", "static", "rcr"):
        st = s.get(name, {})
        lines.append(
            f"| {name} | {st.get('avg_context_tokens', '—')} | "
            f"{st.get('avg_prompt_tokens', '—')} | {st.get('avg_gold_f1', '—')} | "
            f"{st.get('avg_support_recall', '—')} | {st.get('avg_lexical_quality', '—')} |"
        )
    lines += [
        "",
        "### Savings vs Full",
        "",
    ]
    for name in ("static", "rcr"):
        sv = sav.get(name, {})
        lines.append(
            f"- **{name}:** context −{sv.get('context_token_reduction_pct', '—')}%, "
            f"prompt −{sv.get('prompt_token_reduction_pct', '—')}%, "
            f"gold F1 Δ {sv.get('gold_f1_delta', '—')}"
        )
    rcr_pct = rcr_save.get("context_token_reduction_pct")
    verdict = (
        f"RCR reduced average routed context tokens by **{rcr_pct}%** vs Full "
        f"while gold F1 Δ was **{rcr_save.get('gold_f1_delta')}** "
        f"(positive/near-zero ⇒ quality maintained or improved)."
        if rcr_pct is not None
        else "RCR savings not computed."
    )
    lines += [
        "",
        f"**Verdict:** {verdict}",
        "",
        "Paper claim of *up to 30%* token reduction: "
        + (
            f"this run measured **{rcr_pct}%** context-token reduction "
            f"({'meets or exceeds' if (rcr_pct or 0) >= 30 else 'below'} the 30% headline on this fixture set)."
            if rcr_pct is not None
            else "n/a"
        ),
        "",
        "## By benchmark family",
        "",
    ]
    for bname, bsum in bench.items():
        lines.append(f"### {bname}")
        lines.append("")
        bs = bsum["by_strategy"]
        bsv = bsum.get("savings_vs_full", {})
        lines.append(
            f"| Strategy | Ctx tokens | Prompt | Gold F1 | Support recall |"
        )
        lines.append("|----------|----------:|-------:|--------:|---------------:|")
        for name in ("full", "static", "rcr"):
            st = bs.get(name, {})
            lines.append(
                f"| {name} | {st.get('avg_context_tokens', '—')} | "
                f"{st.get('avg_prompt_tokens', '—')} | {st.get('avg_gold_f1', '—')} | "
                f"{st.get('avg_support_recall', '—')} |"
            )
        r = bsv.get("rcr", {})
        lines.append("")
        lines.append(
            f"RCR vs Full: context −{r.get('context_token_reduction_pct', '—')}%, "
            f"F1 Δ {r.get('gold_f1_delta', '—')}."
        )
        lines.append("")

    lines += [
        "## How to reproduce",
        "",
        "```bash",
        "python scripts/run_multihop_experiment.py",
        "# optional: --live-llm  (uses CAIIS / LiteLLM instead of experiment mock)",
        "```",
        "",
        f"Artifacts: `{DEFAULT_RESULTS.relative_to(ROOT)}`, "
        f"`{path.relative_to(ROOT)}`.",
        "",
        "## Limitations",
        "",
        "- Fixtures are synthetic multi-hop chains inspired by HotPotQA / MuSiQue / "
        "2WikiMultihop; they are not the public benchmark splits.",
        "- Default run uses a deterministic mock LLM so answer quality tracks whether "
        "routing retained supporting evidence (not free-form Nemotron generation).",
        "- Token savings depend on an intentionally large M_t; tiny corpora will not "
        "show Full ≫ RCR (see earlier Acme smoke compare).",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(*, max_rounds: int = 2, live_llm: bool = False) -> dict[str, Any]:
    questions, by_id, distractors = load_pack()
    gold_by_query = {q["query"]: q["gold_answer"] for q in questions}
    support_terms = {
        q["query"]: _terms(
            " ".join(by_id[sid]["text"] for sid in q["support_ids"] if sid in by_id)
        )
        for q in questions
    }
    # Keep distinctive terms only
    support_terms = {
        k: {t for t in v if len(t) >= 5} or v for k, v in support_terms.items()
    }

    if live_llm:
        orch = build_orchestrator(use_opensearch=False, force_mock_llm=False)
    else:
        exp_llm = ExperimentMockLLM(gold_by_query, support_terms)
        orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
        # AgentRunner captures llm at construct time — keep both in sync.
        orch.llm = exp_llm
        orch.agents.llm = exp_llm
        orch.task_classifier.llm = exp_llm
        orch.activation.llm = exp_llm

    seed_knowledge(orch, by_id)

    rows: list[dict] = []
    prefill_tokens: list[int] = []

    for q in questions:
        supports = [by_id[sid] for sid in q["support_ids"] if sid in by_id]
        for strategy in (RoutingStrategy.FULL, RoutingStrategy.STATIC, RoutingStrategy.RCR):
            session_id = str(uuid.uuid4())
            pref = prefill_working_memory(
                orch,
                session_id=session_id,
                support_ids=q["support_ids"],
                by_id=by_id,
                distractors=distractors,
            )
            prefill_tokens.append(pref)
            result = orch.run(
                q["query"],
                strategy=strategy,
                max_rounds=max_rounds,
                session_id=session_id,
                knowledge_writeback=False,
                task_model_routing=False,
            )
            metrics = summarize_run(result, q["query"]).to_dict()
            # Collect all routed context texts across rounds
            ctx_texts: list[str] = []
            for t in result.traces:
                ctx_texts.extend(t.context_texts or [])
            g_f1 = token_f1(result.answer, q["gold_answer"])
            s_rec = support_recall(ctx_texts, supports)
            row = {
                **metrics,
                "question_id": q["id"],
                "benchmark": q["benchmark"],
                "query": q["query"],
                "gold_answer": q["gold_answer"],
                "answer": result.answer,
                "gold_f1": round(g_f1, 4),
                "support_recall": round(s_rec, 4),
                "prefill_tokens": pref,
                "n_traces": len(result.traces),
            }
            rows.append(row)

    summary = aggregate(rows)
    bench = by_benchmark(rows)
    meta = {
        "n_questions": len(questions),
        "n_distractors": len(distractors),
        "benchmarks": ", ".join(sorted({q["benchmark"] for q in questions})),
        "avg_prefill_tokens": round(sum(prefill_tokens) / max(1, len(prefill_tokens))),
        "budget_preset": os.environ.get("RCR_BUDGET_PRESET", "savings"),
        "token_backend": token_backend(),
        "max_rounds": max_rounds,
        "live_llm": live_llm,
        "citation": "Liu et al., RCR-Router, arXiv:2508.04903v3",
    }
    return {"meta": meta, "summary": summary, "by_benchmark": bench, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-hop Full/Static/RCR experiment")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Use real LLM client instead of experiment mock",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    payload = run_experiment(max_rounds=args.max_rounds, live_llm=args.live_llm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.no_report:
        write_report(
            args.report,
            rows=payload["rows"],
            summary=payload["summary"],
            bench=payload["by_benchmark"],
            meta=payload["meta"],
        )

    s = payload["summary"]["by_strategy"]
    sav = payload["summary"].get("savings_vs_full", {})
    print(
        f"{'strategy':8} {'ctx':>8} {'prompt':>8} {'goldF1':>8} {'supRec':>8} {'lexQ':>6}"
    )
    for name in ("full", "static", "rcr"):
        st = s[name]
        print(
            f"{name:8} {st['avg_context_tokens']:8.1f} {st['avg_prompt_tokens']:8.1f} "
            f"{st['avg_gold_f1']:8.4f} {st['avg_support_recall']:8.4f} "
            f"{st['avg_lexical_quality']:6.2f}"
        )
    if "rcr" in sav:
        print(
            f"\nRCR vs Full: context −{sav['rcr']['context_token_reduction_pct']}% "
            f"| prompt −{sav['rcr']['prompt_token_reduction_pct']}% "
            f"| F1 Δ {sav['rcr']['gold_f1_delta']}"
        )
    print(f"Wrote {args.out}")
    if not args.no_report:
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
