from __future__ import annotations

from rcr_router.agents import clean_agent_text
from rcr_router.llm import LiteLLMClient


SAMPLE_COT = """
We need to list prime numbers up to 752. Provide answer with primes.
Let's compute primes up to 752.
Check prime 2? yes.
Check prime 3? yes.
Check prime 5? yes.
All good.
Proceed.

The prime numbers from 1 through 752 are:

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751.
"""


def test_strips_nemotron_cot_keeps_prime_list():
    cleaned = LiteLLMClient.clean_completion_text(SAMPLE_COT)
    assert "Check prime" not in cleaned
    assert "Let's compute" not in cleaned
    assert "2, 3, 5, 7" in cleaned
    assert "751" in cleaned
    assert cleaned.index("2, 3, 5") < cleaned.index("751")
    assert cleaned.startswith("2,") or "2, 3, 5" in cleaned


SAMPLE_COT_643 = """
We can manually compute known primes.

Start: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47

We must ensure we didn't miss any: Let's check known prime tables.

Up to 100: list standard: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97 correct.

Now produce final answer as a list maybe comma separated. Provide only final answer. No reasoning.

Thus final answer: list of primes up to 643.

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643
"""


def test_strips_thus_final_answer_primes_643():
    cleaned = LiteLLMClient.clean_completion_text(SAMPLE_COT_643)
    assert "We can manually" not in cleaned
    assert "We must ensure" not in cleaned
    assert "Up to 100" not in cleaned
    assert "Thus final answer" not in cleaned
    assert cleaned.strip().startswith("2,")
    assert cleaned.strip().endswith("643")
    assert "641, 643" in cleaned or "641,643" in cleaned.replace(" ", "")
