"""Run metrics and lightweight answer-quality heuristics."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from .orchestrator import RunResult

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
    "how",
    "do",
    "you",
    "i",
    "want",
    "very",
    "what",
    "which",
    "when",
    "where",
    "who",
    "why",
}


@dataclass
class RunMetrics:
    strategy: str
    llm_tokens: int
    prompt_tokens: int
    completion_tokens: int
    context_tokens: int
    llm_calls: int
    elapsed_sec: float
    answer_quality: float
    answer_quality_notes: str
    token_backend: str = ""
    budget_preset: str = ""
    lexical_quality: float = 0.0
    judge_quality: float | None = None
    judge_justification: str = ""
    judge_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def context_token_total(result: RunResult) -> int:
    return sum(t.context_tokens for t in result.traces)


def _content_terms(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in _STOP
    }


def _numeric_list_quality(query: str, answer: str) -> tuple[float, str] | None:
    """Score comma-separated number lists (e.g. primes up to N) fairly.

    Pure digit lists share no lexical terms with 'prime numbers', so the default
    coverage heuristic wrongly returns ~1.0 for correct answers.
    """
    nums = [int(x) for x in re.findall(r"\b\d+\b", answer)]
    if len(nums) < 8:
        return None
    digitish = sum(1 for c in answer if c.isdigit() or c in ",; \n\t")
    if digitish / max(1, len(answer)) < 0.65:
        return None

    q = (query or "").lower()
    if not any(k in q for k in ("prime", "number", "list", "up to", "through", "≤", "<=")):
        return None
    bounds = [int(x) for x in re.findall(r"\b\d{2,4}\b", query)]
    if not bounds:
        return None
    bound = max(bounds)

    if nums[0] not in (1, 2):
        return None
    over = max(nums) > bound
    mono = all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))
    expected = bound / max(2.0, math.log(bound))
    ratio = len(nums) / expected
    density = 1.0 if 0.45 <= ratio <= 1.6 else 0.45
    mono_s = 1.0 if mono else 0.25
    end_ok = 1.0 if (not over and max(nums) >= bound * 0.85) else 0.35
    bound_ok = 0.0 if over else 1.0
    raw = 1.2 + 1.2 * density + 0.8 * mono_s + 0.6 * end_ok + 0.6 * bound_ok
    score = max(1.0, min(5.0, raw))
    notes = (
        f"numeric_list n={len(nums)} bound={bound} "
        f"max={max(nums)} density={ratio:.2f} mono={mono}"
    )
    return round(score, 2), notes


def lexical_answer_quality(
    query: str, answer: str, evidence_texts: list[str] | None = None
) -> tuple[float, str]:
    """Heuristic quality in [1, 5] without calling an LLM judge.

    Signals: non-empty answer, query term coverage, evidence overlap.
    Special-case: dense numeric lists for 'primes/numbers up to N' queries.
    """
    answer = (answer or "").strip()
    if not answer:
        return 1.0, "Empty answer"
    # Citation-only answers are not useful
    if re.fullmatch(r"(?:\s*\[\s*\d+\s*\]\s*)+", answer):
        return 1.0, "Citation-only answer"

    list_q = _numeric_list_quality(query, answer)
    if list_q is not None:
        return list_q

    q_terms = _content_terms(query)
    a_terms = _content_terms(answer)
    # Digits in the query bound also count toward coverage when present in the answer
    for n in re.findall(r"\b\d{2,4}\b", query):
        q_terms.add(n)
    coverage = (len(q_terms & a_terms) / len(q_terms)) if q_terms else 0.0

    evidence_overlap = 0.0
    if evidence_texts:
        e_terms: set[str] = set()
        for t in evidence_texts:
            e_terms |= _content_terms(t)
            e_terms |= set(re.findall(r"\b\d+\b", t))
        a_all = a_terms | set(re.findall(r"\b\d+\b", answer))
        evidence_overlap = (
            (len(a_all & e_terms) / max(1, len(a_all))) if a_all else 0.0
        )

    length_bonus = 0.15 if 20 <= len(answer.split()) <= 400 else 0.0
    raw = 1.0 + 2.0 * coverage + 1.5 * evidence_overlap + length_bonus
    score = max(1.0, min(5.0, raw))
    notes = f"coverage={coverage:.2f} evidence_overlap={evidence_overlap:.2f}"
    return round(score, 2), notes


def summarize_run(
    result: RunResult,
    query: str,
    *,
    llm: Any | None = None,
    use_judge: bool | None = None,
) -> RunMetrics:
    """Summarize a run. Lexical quality is always computed (fast).

    Optional LLM judge (paper Answer Quality Score) runs only when
    ``use_judge`` is true or ``RCR_LLM_JUDGE=1`` and an ``llm`` client is passed.
    """
    from .answer_judge import judge_enabled, llm_answer_quality

    # Prefer Searcher outputs + any routed knowledge/fact texts (not Acme-hardcoded).
    evidence: list[str] = []
    for t in result.traces:
        if t.role == "searcher" and t.output:
            evidence.append(t.output)
        for txt in t.context_texts:
            if not txt:
                continue
            lower = txt.lower()
            if lower.startswith("user_query:"):
                continue
            if (
                "fact" in lower
                or "answer" in lower
                or "ingredient" in lower
                or "recipe" in lower
                or "evidence" in lower
                or "prime" in lower
                or "know-" in (getattr(t, "context_ids", None) or [""])[0]
                or len(txt.split()) >= 8
            ):
                evidence.append(txt)
    lexical, notes = lexical_answer_quality(query, result.answer, evidence)
    enabled = judge_enabled() if use_judge is None else bool(use_judge)
    judge_score: float | None = None
    judge_just = ""
    if enabled and llm is not None:
        jr = llm_answer_quality(query, result.answer, llm, evidence=evidence)
        if jr.ok:
            judge_score = jr.score
            judge_just = jr.justification
            notes = f"{notes}; judge={jr.score}"
        else:
            notes = f"{notes}; judge_error={jr.error or 'failed'}"

    usage = result.usage or {}
    budget = getattr(result, "budget", None) or {}
    return RunMetrics(
        strategy=result.strategy,
        llm_tokens=int(usage.get("total_tokens", 0)),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        context_tokens=context_token_total(result),
        llm_calls=int(usage.get("calls", 0)),
        elapsed_sec=round(result.elapsed_sec, 4),
        # Primary column stays lexical so default smoke/compare stays fast & stable.
        answer_quality=lexical,
        answer_quality_notes=notes if not judge_just else f"{notes} | {judge_just}",
        token_backend=str(getattr(result, "token_backend", "") or ""),
        budget_preset=str(budget.get("preset") or ""),
        lexical_quality=lexical,
        judge_quality=judge_score,
        judge_justification=judge_just,
        judge_enabled=enabled and llm is not None,
    )
