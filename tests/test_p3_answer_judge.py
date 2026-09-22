"""P3: optional LLM Answer Quality Score (paper Appendix A.4)."""

from __future__ import annotations

from rcr_router.answer_judge import (
    build_judge_prompt,
    judge_enabled,
    llm_answer_quality,
    parse_judge_response,
)
from rcr_router.factory import build_orchestrator
from rcr_router.llm import MockLLMClient
from rcr_router.metrics import summarize_run
from rcr_router.models import RoutingStrategy


def test_parse_judge_json():
    r = parse_judge_response(
        '{"score": 5, "justification": "Complete and correct."}'
    )
    assert r.ok and r.score == 5.0
    assert "Complete" in r.justification


def test_parse_judge_fenced_and_clamped():
    r = parse_judge_response('```json\n{"score": 9, "justification": "too high"}\n```')
    assert r.ok and r.score == 5.0


def test_parse_judge_fallback_number():
    r = parse_judge_response("I would give this a 3 overall.")
    assert r.ok and r.score == 3.0


def test_build_prompt_matches_paper_shape():
    p = build_judge_prompt("What drove growth?", "Cloud services.")
    assert "User Query:" in p
    assert "Answer:" in p
    assert '"score"' in p
    assert "justification" in p


def test_mock_llm_judge():
    llm = MockLLMClient()
    jr = llm_answer_quality(
        "What drove Acme Corp revenue growth in Q3?",
        "Cloud subscription services drove growth.",
        llm,
    )
    assert jr.ok
    assert 1.0 <= jr.score <= 5.0
    assert jr.justification


def test_summarize_run_default_skips_judge():
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    result = orch.run(
        "What drove Acme Corp revenue growth in Q3?",
        strategy=RoutingStrategy.RCR,
        max_rounds=1,
        knowledge_writeback=False,
    )
    m = summarize_run(result, "What drove Acme Corp revenue growth in Q3?")
    assert m.judge_quality is None
    assert m.judge_enabled is False
    assert 1.0 <= m.answer_quality <= 5.0
    assert m.lexical_quality == m.answer_quality


def test_summarize_run_with_judge(monkeypatch):
    monkeypatch.delenv("RCR_LLM_JUDGE", raising=False)
    orch = build_orchestrator(use_opensearch=False, force_mock_llm=True)
    q = "What drove Acme Corp revenue growth in Q3?"
    result = orch.run(q, strategy=RoutingStrategy.RCR, max_rounds=1, knowledge_writeback=False)
    m = summarize_run(result, q, llm=orch.llm, use_judge=True)
    assert m.judge_enabled is True
    assert m.judge_quality == 4.0
    assert m.lexical_quality == m.answer_quality
    assert m.judge_justification


def test_judge_enabled_env(monkeypatch):
    monkeypatch.setenv("RCR_LLM_JUDGE", "1")
    assert judge_enabled() is True
    monkeypatch.setenv("RCR_LLM_JUDGE", "0")
    assert judge_enabled() is False
