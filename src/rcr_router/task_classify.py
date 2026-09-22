"""Classify an incoming user prompt into a shared task taxonomy."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from .endpoint_store import INTENDED_TASKS
from .llm import LLMClient, MockLLMClient

# Same vocabulary as model intended_task (Settings).
QUERY_TASKS: tuple[str, ...] = INTENDED_TASKS

TASK_HEURISTICS: dict[str, tuple[str, ...]] = {
    "coding": (
        "code",
        "coding",
        "function",
        "bug",
        "stack trace",
        "implement",
        "refactor",
        "compile",
        "python",
        "javascript",
        "api endpoint",
        "unit test",
        "pull request",
        "dockerfile",
    ),
    "reasoning": (
        "why did",
        "reason about",
        "decompose",
        "multi-step",
        "strategy",
        "plan how",
        "think through",
        "root cause",
        "trade-off",
        "tradeoff",
    ),
    "extraction": (
        "extract",
        "find evidence",
        "retrieve",
        "look up",
        "from the passage",
        "from the document",
        "factual claims",
        "what facts",
        "cite",
        "grounded in",
    ),
    "synthesis": (
        "summarize",
        "summary",
        "recommend",
        "recommendation",
        "conclude",
        "overall",
        "in one sentence",
        "what drove",
        "key takeaway",
        "tl;dr",
        "tldr",
    ),
    "routing": (
        "which agent",
        "route this",
        "classify this query",
        "which model",
        "who should answer",
    ),
}


@dataclass
class TaskDecision:
    task: str = "generic"
    confidence: float = 0.0
    source: str = "heuristic"  # heuristic | llm | fallback
    rationale: str = ""

    def normalized(self) -> "TaskDecision":
        task = (self.task or "generic").strip().lower()
        if task not in QUERY_TASKS:
            task = "generic"
        conf = max(0.0, min(1.0, float(self.confidence)))
        return TaskDecision(task=task, confidence=conf, source=self.source, rationale=self.rationale)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())


def heuristic_classify(query: str) -> TaskDecision:
    """Keyword / phrase prior — no LLM call."""
    q = (query or "").lower()
    scores: dict[str, float] = {t: 0.0 for t in QUERY_TASKS if t != "generic"}
    notes: list[str] = []
    for task, phrases in TASK_HEURISTICS.items():
        hits = [p for p in phrases if p in q]
        if hits:
            scores[task] = float(len(hits))
            notes.append(f"{task}:{','.join(hits[:3])}")
    if not any(scores.values()):
        return TaskDecision(task="generic", confidence=0.35, source="heuristic", rationale="no keyword hits")
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    confidence = min(0.95, 0.45 + 0.2 * scores[best] + 0.15 * (scores[best] / total))
    return TaskDecision(
        task=best,
        confidence=round(confidence, 3),
        source="heuristic",
        rationale="; ".join(notes) or best,
    ).normalized()


def _parse_llm_task(text: str) -> TaskDecision | None:
    cleaned = text.strip()
    # Strip fences / reasoning fluff
    fence = re.search(r"\{[^{}]*\}", cleaned, flags=re.S)
    if not fence:
        return None
    try:
        data = json.loads(fence.group(0))
    except json.JSONDecodeError:
        return None
    task = str(data.get("task") or "generic").lower()
    conf = float(data.get("confidence") or 0.5)
    rationale = str(data.get("rationale") or "llm")
    return TaskDecision(task=task, confidence=conf, source="llm", rationale=rationale).normalized()


def llm_classify(query: str, llm: LLMClient) -> TaskDecision:
    tasks = ", ".join(QUERY_TASKS)
    resp = llm.complete(
        [
            {
                "role": "system",
                "content": (
                    "You classify user prompts into exactly one task label. "
                    "Output JSON only: "
                    '{"task":"<label>","confidence":0.0-1.0,"rationale":"<short>"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Allowed labels: [{tasks}].\n"
                    f'Classify this prompt:\n"""{query[:800]}"""'
                ),
            },
        ]
    )
    parsed = _parse_llm_task(resp.text or "")
    if parsed is None:
        return TaskDecision(
            task="generic",
            confidence=0.2,
            source="fallback",
            rationale=f"unparseable llm: {(resp.text or '')[:120]}",
        )
    return parsed


class TaskClassifier:
    """Heuristic-first classifier; optional LLM refinement via RCR_TASK_CLASSIFIER."""

    def __init__(self, llm: LLMClient | None = None, mode: str | None = None) -> None:
        self.llm = llm
        self.mode = (mode or os.getenv("RCR_TASK_CLASSIFIER", "heuristic")).strip().lower()

    def classify(self, query: str) -> TaskDecision:
        if self.mode in {"off", "none", "disabled"}:
            return TaskDecision(task="generic", confidence=0.0, source="fallback", rationale="classifier off")

        base = heuristic_classify(query)
        if self.mode in {"heuristic", "rules"}:
            return base

        if self.mode in {"llm", "hybrid"} and self.llm is not None:
            # Hybrid: trust LLM when heuristics are weak or disagree after parse
            if self.mode == "llm" or base.confidence < 0.6 or base.task == "generic":
                try:
                    llm_dec = llm_classify(query, self.llm)
                    if self.mode == "hybrid" and base.task != "generic" and llm_dec.task == "generic":
                        return base
                    return llm_dec
                except Exception as exc:  # noqa: BLE001
                    base.rationale = f"{base.rationale}; llm_failed:{exc}"
                    return base
            return base

        return base


def task_match_score(item_task: str | None, query_task: str | None) -> float:
    """0..1 soft match for importance scoring / retrieval boost."""
    if not query_task or query_task == "generic":
        return 0.5  # neutral
    it = (item_task or "").strip().lower() or None
    if it is None:
        return 0.45  # untagged corpus — slight soft penalty vs exact match
    if it == query_task:
        return 1.0
    if it == "generic":
        return 0.55
    return 0.2
