"""Structured memory extraction (paper Memory Update: extract → structure).

Deterministic, role-aware parsing — no LLM required. Produces YAML-friendly
units with stable ``struct_key`` values for conflict resolution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

from .models import AgentRole, DocType

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_PLAN_HDR_RE = re.compile(r"^\s*(?:plan|steps?|sub-?goals?)\s*[:\-]\s*", re.I)
_FACT_HDR_RE = re.compile(r"^\s*(?:fact|evidence|finding)\s*[:\-]\s*", re.I)
_KNOW_ID_RE = re.compile(r"\b(know-[a-zA-Z0-9\-]+)\b")
_CITE_RE = re.compile(r"\[\s*(\d+)\s*\]")


@dataclass
class StructuredUnit:
    kind: str  # fact | plan_step | evidence | answer | interaction
    key: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    doc_type: str = DocType.INTERACTION.value


def slug_key(text: str, *, max_words: int = 8) -> str:
    words = _SLUG_RE.sub(" ", (text or "").lower()).split()
    stem = "-".join(words[:max_words]) if words else "empty"
    return stem[:80] or "empty"


def fact_slug(text: str) -> str:
    """Slug for fact keys — strip digits/% so numeric revisions share a key."""
    normalized = re.sub(r"\d+(?:[.,]\d+)?%?", " ", text or "")
    return slug_key(normalized, max_words=6)


def to_yaml_block(unit: StructuredUnit) -> str:
    """Debug/export YAML — not used for hot-path memory ``text`` (too redundant)."""
    body = {
        "kind": unit.kind,
        "key": unit.key,
        **unit.payload,
        "text": unit.text,
    }
    return yaml.safe_dump(body, sort_keys=False, allow_unicode=True).strip()


def to_memory_text(unit: StructuredUnit) -> str:
    """Lean text for working-memory / prompts — content only; keys live on the item."""
    body = (unit.text or "").strip()
    if unit.kind == "plan_step":
        step = unit.payload.get("step")
        if step is not None:
            return f"{step}. {body}"
        return body
    if unit.kind == "evidence":
        return body
    return body


_LOW_VALUE_PLAN = re.compile(
    r"(?i)^\s*("
    r"thus\s+output|"
    r"output\s+(?:something|only|the\s+plan)|"
    r"we\s+should\s+only|"
    r"no\s+extra\s+text|"
    r"no\s+chain[- ]of[- ]thought|"
    r"present\s+the\s+results?\.?|"
    r"be\s+concise|"
    r"follow(?:ing)?\s+instructions|"
    r"do\s+not\s+include|"
    r"remember[:\s]|"
    r"example[:\s]|"
    r"something\s+like\s*:"
    r").*$"
)


def is_low_value_plan_step(text: str) -> bool:
    """Drop planner meta/instruction leakage that is not a real search step."""
    t = (text or "").strip()
    if len(t) < 12:
        return True
    if _LOW_VALUE_PLAN.match(t):
        return True
    # Bare imperatives with no content noun phrase
    lower = t.lower().rstrip(".")
    if lower in {
        "present the results",
        "summarize",
        "search",
        "retrieve",
        "find information",
        "output the plan",
    }:
        return True
    return False


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _split_units(text: str) -> list[str]:
    """Split agent output into candidate atomic strings."""
    chunks: list[str] = []
    for ln in _lines(text):
        ln = _PLAN_HDR_RE.sub("", ln)
        ln = _FACT_HDR_RE.sub("", ln)
        ln = _BULLET_RE.sub("", ln).strip()
        if len(ln) < 8:
            continue
        # Drop pure citation chrome
        if re.fullmatch(r"(?:\[\s*\d+\s*\]\s*)+", ln):
            continue
        chunks.append(ln)
    if chunks:
        return chunks
    cleaned = (text or "").strip()
    if len(cleaned) >= 8:
        return [cleaned]
    return []


def _plan_near_dupe(a: str, b: str, *, thresh: float = 0.5) -> bool:
    """True when two plan lines are paraphrases of the same step."""
    wa = set(_SLUG_RE.sub(" ", (a or "").lower()).split())
    wb = set(_SLUG_RE.sub(" ", (b or "").lower()).split())
    # Drop ultra-common planning verbs so "search X" ≈ "find X" ≈ "retrieve X"
    stop = {
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "and",
        "or",
        "by",
        "search",
        "find",
        "retrieve",
        "identify",
        "list",
        "get",
        "look",
        "about",
        "information",
        "names",
        "name",
        "their",
        "intents",
    }
    wa -= stop
    wb -= stop
    if not wa or not wb:
        return False
    overlap = len(wa & wb) / max(1, min(len(wa), len(wb)))
    return overlap >= thresh


def extract_evidence_ids(text: str) -> list[str]:
    ids = list(dict.fromkeys(_KNOW_ID_RE.findall(text or "")))
    return ids


def extract_structured_units(
    text: str,
    *,
    role: AgentRole,
    max_units: int = 8,
) -> list[StructuredUnit]:
    """Extract structured memory units from an agent LLM output."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    evidence_ids = extract_evidence_ids(cleaned)
    units: list[StructuredUnit] = []

    if role == AgentRole.PLANNER:
        step_n = 0
        for step in _split_units(cleaned)[:max_units]:
            if is_low_value_plan_step(step):
                continue
            # Drop paraphrased duplicates (e.g. "search moons" / "find moons")
            if any(
                _plan_near_dupe(step, u.text) for u in units if u.kind == "plan_step"
            ):
                continue
            step_n += 1
            units.append(
                StructuredUnit(
                    kind="plan_step",
                    key=f"plan_step:{step_n}",
                    text=step,
                    payload={"step": step_n, "evidence_ids": evidence_ids},
                    doc_type=DocType.PLAN.value,
                )
            )
    elif role == AgentRole.SEARCHER:
        for chunk in _split_units(cleaned)[:max_units]:
            key = f"fact:{fact_slug(chunk)}"
            units.append(
                StructuredUnit(
                    kind="fact",
                    key=key,
                    text=chunk,
                    payload={"evidence_ids": evidence_ids or extract_evidence_ids(chunk)},
                    doc_type=DocType.FACT.value,
                )
            )
        for eid in evidence_ids:
            units.append(
                StructuredUnit(
                    kind="evidence",
                    key=f"evidence:{eid}",
                    text=f"Evidence ref: {eid}",
                    payload={"evidence_id": eid},
                    doc_type=DocType.FACT.value,
                )
            )
    elif role == AgentRole.RECOMMENDER:
        # One canonical answer; collapse whitespace for storage text
        answer = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        units.append(
            StructuredUnit(
                kind="answer",
                key="answer:final",
                text=answer,
                payload={"evidence_ids": evidence_ids, "cites": _CITE_RE.findall(cleaned)},
                doc_type=DocType.INTERACTION.value,
            )
        )
    else:
        units.append(
            StructuredUnit(
                kind="interaction",
                key=f"interaction:{slug_key(cleaned, max_words=6)}",
                text=cleaned,
                payload={},
                doc_type=DocType.INTERACTION.value,
            )
        )

    # Deduplicate by key within this extraction (last wins)
    by_key: dict[str, StructuredUnit] = {}
    for u in units:
        by_key[u.key] = u
    return list(by_key.values())[: max_units + len(evidence_ids)]
