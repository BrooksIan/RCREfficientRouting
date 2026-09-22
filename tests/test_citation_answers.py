from __future__ import annotations

from rcr_router.agents import (
    is_citation_only,
    resolve_citations_to_context,
    synthesize_answer_from_context,
)
from rcr_router.models import AgentRole, MemoryItem
from rcr_router.metrics import lexical_answer_quality


def test_is_citation_only():
    assert is_citation_only("[2] [3]")
    assert is_citation_only(" [1]  [2] ")
    assert not is_citation_only(
        "NY Style pizza dough needs 500g flour and 320g water [2] [3]."
    )


def test_resolve_citations_expands_snippets():
    items = [
        MemoryItem(text="USER_QUERY: pizza?"),
        MemoryItem(text="500g bread flour, 320g water, 10g salt"),
        MemoryItem(text="Knead 10 minutes, cold ferment 24h"),
    ]
    out = resolve_citations_to_context("[2] [3]", items)
    assert "500g bread flour" in out
    assert "Knead 10 minutes" in out
    assert "USER_QUERY" not in out


def test_synthesize_and_quality_reject_citations():
    items = [
        MemoryItem(text="USER_QUERY: how to make dough"),
        MemoryItem(text="Mix flour water yeast salt and oil for NY pizza dough"),
    ]
    synth = synthesize_answer_from_context(items, "how to make NY pizza dough")
    assert "flour" in synth.lower()
    score, notes = lexical_answer_quality("pizza dough", "[2] [3]")
    assert score == 1.0
    assert "Citation-only" in notes


def test_numeric_list_primes_scores_well():
    q = "What are the prime numbers of up to 502?"
    a = (
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, "
        "73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, "
        "157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, "
        "239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, "
        "331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, "
        "421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499"
    )
    score, notes = lexical_answer_quality(q, a)
    assert score >= 3.5
    assert "numeric_list" in notes
