"""P2: structured memory update — extract, keys, conflict resolve."""

from __future__ import annotations

from rcr_router.memory_store_memory import InMemoryMemoryStore
from rcr_router.memory_update import MemoryUpdater
from rcr_router.models import AgentRole, TaskStage
from rcr_router.structured_memory import extract_structured_units, to_yaml_block


def test_extract_plan_steps():
    units = extract_structured_units(
        "1. Decompose the revenue question\n2. Retrieve cloud metrics\n3. Summarize drivers",
        role=AgentRole.PLANNER,
    )
    assert len(units) == 3
    assert units[0].kind == "plan_step"
    assert units[0].key == "plan_step:1"
    assert units[2].key == "plan_step:3"
    assert "Decompose" in units[0].text


def test_extract_facts_and_evidence_ids():
    text = (
        "- Acme Corp reported cloud-driven revenue growth in Q3 (know-acme-q3).\n"
        "- Operating margin improved year over year."
    )
    units = extract_structured_units(text, role=AgentRole.SEARCHER)
    kinds = {u.kind for u in units}
    assert "fact" in kinds
    assert "evidence" in kinds
    assert any(u.key.startswith("evidence:know-") for u in units)


def test_extract_answer_key():
    units = extract_structured_units(
        "Cloud growth was the primary revenue driver in Q3.",
        role=AgentRole.RECOMMENDER,
    )
    assert len(units) == 1
    assert units[0].key == "answer:final"
    assert units[0].kind == "answer"


def test_yaml_block_roundtrip_shape():
    units = extract_structured_units(
        "- Fact about pizza dough hydration at 65%.",
        role=AgentRole.SEARCHER,
    )
    block = to_yaml_block(units[0])
    assert "kind: fact" in block
    assert "key:" in block
    assert "65%" in block


def test_memory_text_is_lean_not_yaml():
    from rcr_router.structured_memory import to_memory_text

    units = extract_structured_units(
        "1. Retrieve nearest stars to the Sun\n2. Present the results.\n3. We should only output the plan, no extra text.",
        role=AgentRole.PLANNER,
    )
    assert len(units) == 1
    assert "nearest stars" in units[0].text.lower()
    lean = to_memory_text(units[0])
    assert "kind:" not in lean
    assert "struct_key" not in lean
    assert lean.startswith("1.")


def test_filters_meta_plan_steps():
    units = extract_structured_units(
        "Plan:\n"
        "Thus output something like:\n"
        "Retrieve list of nearest stars to the Sun.\n"
        "Present the results.\n"
        "We should only output the plan, no extra text.\n",
        role=AgentRole.PLANNER,
    )
    assert len(units) == 1
    assert "nearest" in units[0].text.lower()

def test_conflict_replace_same_fact_key():
    mem = InMemoryMemoryStore()
    updater = MemoryUpdater(mem)
    session = "p2-conflict"
    first = updater.update_from_agent_output(
        text="- Revenue grew 10% in cloud segment.",
        role=AgentRole.SEARCHER,
        stage=TaskStage.SEARCHING,
        session_id=session,
        round_idx=1,
    )
    assert len(first) == 1
    key = first[0].struct_key
    assert key and key.startswith("fact:")

    # Same topic, revised number → same fact key → replace
    revised = updater.update_from_agent_output(
        text="- Revenue grew 22% in cloud segment.",
        role=AgentRole.SEARCHER,
        stage=TaskStage.SEARCHING,
        session_id=session,
        round_idx=3,
    )
    assert revised
    assert revised[0].struct_key == key
    facts = updater.list_by_kind(session, "fact")
    assert len(facts) == 1
    assert "22%" in facts[0].text

    # Answer key always replaces
    a1 = updater.update_from_agent_output(
        text="Primary driver was cloud revenue growth.",
        role=AgentRole.RECOMMENDER,
        stage=TaskStage.RECOMMENDING,
        session_id=session,
        round_idx=3,
    )
    a2 = updater.update_from_agent_output(
        text="Primary driver was cloud revenue growth and margin expansion.",
        role=AgentRole.RECOMMENDER,
        stage=TaskStage.RECOMMENDING,
        session_id=session,
        round_idx=4,
    )
    assert a1 and a2
    answers = updater.list_by_kind(session, "answer")
    assert len(answers) == 1
    assert "margin expansion" in answers[0].text
    assert updater.get_by_key(session, "answer:final") is not None


def test_plan_supersession_prunes_extra_steps():
    mem = InMemoryMemoryStore()
    updater = MemoryUpdater(mem)
    session = "p2-plan"
    updater.update_from_agent_output(
        text="1. First gather data\n2. Then analyze trends\n3. Finally draft answer",
        role=AgentRole.PLANNER,
        stage=TaskStage.PLANNING,
        session_id=session,
        round_idx=1,
    )
    assert len(updater.list_by_kind(session, "plan_step")) == 3

    updater.update_from_agent_output(
        text="1. Gather revised metrics\n2. Draft concise answer",
        role=AgentRole.PLANNER,
        stage=TaskStage.PLANNING,
        session_id=session,
        round_idx=2,
    )
    steps = updater.list_by_kind(session, "plan_step")
    assert len(steps) == 2
    assert {s.struct_key for s in steps} == {"plan_step:1", "plan_step:2"}
    assert "revised" in updater.get_by_key(session, "plan_step:1").text  # type: ignore[union-attr]


def test_queryable_by_role_and_key():
    mem = InMemoryMemoryStore()
    updater = MemoryUpdater(mem)
    session = "p2-query"
    updater.update_from_agent_output(
        text="1. Search pizza dough recipes\n2. List hydration ratios",
        role=AgentRole.PLANNER,
        stage=TaskStage.PLANNING,
        session_id=session,
        round_idx=1,
    )
    updater.update_from_agent_output(
        text="- Neapolitan dough uses about 65% hydration.",
        role=AgentRole.SEARCHER,
        stage=TaskStage.SEARCHING,
        session_id=session,
        round_idx=1,
    )
    plans = [m for m in mem.list_session(session) if m.role_tag == "planner"]
    facts = updater.list_by_kind(session, "fact", role_tag="searcher")
    assert plans
    assert facts
    assert all(p.struct_key for p in plans)
    assert all(f.structure.get("kind") == "fact" for f in facts)
