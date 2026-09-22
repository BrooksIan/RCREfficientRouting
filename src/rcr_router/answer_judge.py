"""Optional LLM-as-judge Answer Quality Score (paper Appendix A.4).

Default offline path stays lexical-only. Enable with ``RCR_LLM_JUDGE=1`` or
``summarize_run(..., use_judge=True, llm=...)``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient, LLMResponse

JUDGE_SYSTEM = (
    "You are an expert judge. Your task is to evaluate how well the answer "
    "responds to the user's query. Judge correctness, relevance, completeness, "
    "and clarity. Reply with JSON only — no markdown fences."
)

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class JudgeResult:
    score: float
    justification: str
    raw: str = ""
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "justification": self.justification,
            "ok": self.ok,
            "error": self.error,
        }


def judge_enabled() -> bool:
    return os.getenv("RCR_LLM_JUDGE", "").strip().lower() in {"1", "true", "yes", "on"}


def build_judge_prompt(query: str, answer: str, *, evidence: list[str] | None = None) -> str:
    """Paper-style prompt (Appendix A.4) with optional evidence context."""
    parts = [
        f"User Query:\n{(query or '').strip()}",
        f"Answer:\n{(answer or '').strip()}",
    ]
    if evidence:
        clipped = [e.strip() for e in evidence if e and e.strip()][:6]
        if clipped:
            joined = "\n---\n".join(c[:400] for c in clipped)
            parts.append(f"Supporting evidence (optional):\n{joined}")
    parts.append(
        'Please provide a JSON object with the following format: '
        '{"score": (1 to 5), "justification": "a short explanation of the score"}'
    )
    return "\n\n".join(parts)


def parse_judge_response(text: str) -> JudgeResult:
    raw = (text or "").strip()
    if not raw:
        return JudgeResult(score=1.0, justification="", raw=raw, ok=False, error="empty")

    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    payload: dict[str, Any] | None = None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_RE.search(cleaned)
        if m:
            try:
                payload = json.loads(m.group(0))
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict) or "score" not in payload:
        # Fallback: bare number in text
        nums = re.findall(r"\b([1-5](?:\.\d+)?)\b", cleaned)
        if nums:
            score = max(1.0, min(5.0, float(nums[0])))
            return JudgeResult(
                score=round(score, 2),
                justification=cleaned[:240],
                raw=raw,
                ok=True,
                error="score_fallback",
            )
        return JudgeResult(
            score=1.0,
            justification="",
            raw=raw,
            ok=False,
            error="unparseable",
        )

    try:
        score = float(payload["score"])
    except (TypeError, ValueError):
        return JudgeResult(
            score=1.0,
            justification=str(payload.get("justification") or ""),
            raw=raw,
            ok=False,
            error="bad_score",
        )

    score = max(1.0, min(5.0, score))
    just = str(payload.get("justification") or payload.get("reason") or "").strip()
    return JudgeResult(score=round(score, 2), justification=just, raw=raw, ok=True)


def llm_answer_quality(
    query: str,
    answer: str,
    llm: LLMClient,
    *,
    evidence: list[str] | None = None,
    model: str | None = None,
) -> JudgeResult:
    """Call an LLM judge; returns a 1–5 score + justification."""
    prompt = build_judge_prompt(query, answer, evidence=evidence)
    try:
        resp: LLMResponse = llm.complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=model or os.getenv("RCR_JUDGE_MODEL") or None,
        )
    except Exception as exc:  # noqa: BLE001 — judge must not break compare
        return JudgeResult(
            score=1.0,
            justification="",
            ok=False,
            error=f"llm_error:{type(exc).__name__}",
        )
    return parse_judge_response(resp.text or "")
