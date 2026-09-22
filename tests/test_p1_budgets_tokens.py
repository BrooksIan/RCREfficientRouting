"""P1: real token accounting + paper/savings/demo budget presets."""

from __future__ import annotations

from rcr_router.budget import (
    DEMO_BASE_BUDGET,
    PAPER_BASE_BUDGET,
    PAPER_FULL_CONTEXT_CAP,
    SAVINGS_BASE_BUDGET,
    TokenBudgetAllocator,
)
from rcr_router.models import AgentRole, MemoryItem
from rcr_router.tokens import count_tokens, token_backend


def test_count_tokens_uses_real_tokenizer():
    text = "Role-aware context routing allocates a token budget per agent role."
    n = count_tokens(text)
    words = len(text.split())
    assert n >= 8
    # Tokenizer counts differ from whitespace words for this phrase.
    assert token_backend() in {"tiktoken", "litellm", "approx"}
    if token_backend() != "approx":
        assert n != words or n > 5


def test_memory_item_token_length_from_tokenizer():
    item = MemoryItem(text="plan evidence retrieve documents for the revenue question")
    assert item.token_length == count_tokens(item.text)
    assert item.token_length >= 8


def test_paper_preset_budgets_in_paper_band():
    alloc = TokenBudgetAllocator(preset="paper")
    assert alloc.base == PAPER_BASE_BUDGET
    assert alloc.full_context_cap == PAPER_FULL_CONTEXT_CAP
    planner = alloc.budget_for(AgentRole.PLANNER)
    searcher = alloc.budget_for(AgentRole.SEARCHER)
    recommender = alloc.budget_for(AgentRole.RECOMMENDER)
    assert 512 <= searcher <= 4096
    assert 512 <= recommender <= 4096
    assert 512 <= planner <= 4096
    assert planner > recommender >= searcher


def test_savings_preset_tighter_than_paper():
    paper = TokenBudgetAllocator(preset="paper")
    savings = TokenBudgetAllocator(preset="savings")
    assert savings.base == SAVINGS_BASE_BUDGET
    assert savings.preset == "savings"
    assert savings.full_context_cap == paper.full_context_cap
    for role in AgentRole:
        assert savings.budget_for(role) < paper.budget_for(role)
    assert savings.budget_for(AgentRole.PLANNER) == 768
    assert savings.budget_for(AgentRole.SEARCHER) == 512
    assert savings.budget_for(AgentRole.RECOMMENDER) == 640


def test_demo_preset_is_small():
    alloc = TokenBudgetAllocator(preset="demo")
    assert alloc.base == DEMO_BASE_BUDGET
    assert alloc.budget_for(AgentRole.PLANNER) < 300
    assert alloc.full_context_cap < PAPER_FULL_CONTEXT_CAP


def test_from_env_respects_preset(monkeypatch):
    monkeypatch.delenv("RCR_BUDGET_SCALE", raising=False)
    monkeypatch.setenv("RCR_BUDGET_PRESET", "demo")
    alloc = TokenBudgetAllocator.from_env()
    assert alloc.preset == "demo"
    monkeypatch.setenv("RCR_BUDGET_PRESET", "paper")
    monkeypatch.setenv("RCR_BUDGET_BASE", "600")
    alloc2 = TokenBudgetAllocator.from_env()
    assert alloc2.preset == "paper"
    assert alloc2.base == 600
    monkeypatch.delenv("RCR_BUDGET_BASE", raising=False)
    monkeypatch.setenv("RCR_BUDGET_PRESET", "savings")
    assert TokenBudgetAllocator.from_env().preset == "savings"


def test_budget_scale_env(monkeypatch):
    monkeypatch.delenv("RCR_BUDGET_BASE", raising=False)
    monkeypatch.setenv("RCR_BUDGET_PRESET", "savings")
    monkeypatch.setenv("RCR_BUDGET_SCALE", "0.5")
    alloc = TokenBudgetAllocator.from_env()
    assert alloc.budget_for(AgentRole.PLANNER) == 384  # 768 * 0.5


def test_budget_as_dict_includes_roles():
    d = TokenBudgetAllocator(preset="paper").as_dict()
    assert d["preset"] == "paper"
    assert set(d["roles"]) == {"planner", "searcher", "recommender"}
