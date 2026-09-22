from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .agent_router import AgentActivationRouter
from .agents import AgentRunner, is_citation_only
from .budget import TokenBudgetAllocator
from .knowledge_probe import KnowledgeProbeResult, probe_knowledge
from .llm import LLMClient, MockLLMClient, UsageTracker
from .memory_update import MemoryUpdater
from .model_routing import ModelRegistry, active_model_registry
from .models import (
    STAGE_FOR_ROLE,
    AgentRole,
    DocType,
    MemoryItem,
    RoutingStrategy,
    TaskStage,
)
from .router import RCRRouter
from .scorer import ImportanceScorer
from .store import KnowledgeStore, MemoryStore
from .task_classify import TaskClassifier, TaskDecision
from .task_model_policy import RunModelPlan, build_run_model_plan, task_model_routing_enabled
from .tokens import count_tokens, token_backend
from .knowledge_writeback import (
    KnowledgeWriteback,
    compact_mode,
    schedule_compact_after_prompt,
    select_knowledge_for_context,
)


@dataclass
class RoundTrace:
    round_idx: int
    role: str
    strategy: str
    context_ids: list[str]
    context_texts: list[str]
    output: str
    context_tokens: int
    model: str = ""


@dataclass
class RunResult:
    session_id: str
    answer: str
    strategy: str
    max_rounds: int
    usage: dict[str, Any]
    traces: list[RoundTrace] = field(default_factory=list)
    memory_snapshot: list[dict[str, Any]] = field(default_factory=list)
    model_profile: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    knowledge_writebacks: list[dict[str, Any]] = field(default_factory=list)
    knowledge_compaction: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    token_backend: str = ""
    knowledge_probe: dict[str, Any] = field(default_factory=dict)
    model_plan: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0


class Orchestrator:
    def __init__(
        self,
        memory: MemoryStore,
        knowledge: KnowledgeStore,
        llm: LLMClient | None = None,
        budget: TokenBudgetAllocator | None = None,
        scorer: ImportanceScorer | None = None,
        registry: ModelRegistry | None = None,
        task_classifier: TaskClassifier | None = None,
    ) -> None:
        self.memory = memory
        self.knowledge = knowledge
        self.llm = llm or MockLLMClient()
        self.registry = registry or active_model_registry()
        self.budget = budget or TokenBudgetAllocator.from_env()
        self.router = RCRRouter(memory, budget=self.budget, scorer=scorer)
        self.updater = MemoryUpdater(memory)
        self.agents = AgentRunner(self.llm, registry=self.registry)
        self.activation = AgentActivationRouter(self.llm, registry=self.registry)
        self.task_classifier = task_classifier or TaskClassifier(llm=self.llm)
        self.writeback = KnowledgeWriteback(knowledge)

    def run(
        self,
        query: str,
        *,
        strategy: RoutingStrategy | str = RoutingStrategy.RCR,
        max_rounds: int = 3,
        session_id: str | None = None,
        model: str | None = None,
        knowledge_writeback: bool | None = None,
        task_model_routing: bool | None = None,
    ) -> RunResult:
        started = time.time()
        session_id = session_id or str(uuid.uuid4())
        strategy_enum = (
            strategy if isinstance(strategy, RoutingStrategy) else RoutingStrategy(strategy)
        )
        usage = UsageTracker()
        traces: list[RoundTrace] = []
        prior_roles: list[str] = []
        self.writeback.writes.clear()
        prev_wb = self.writeback.enabled
        if knowledge_writeback is not None:
            self.writeback.enabled = knowledge_writeback

        probe_result: KnowledgeProbeResult | None = None
        model_plan = RunModelPlan(
            enabled=False,
            task="generic",
            knowledge_strength="weak",
            note="not_run",
        )
        probed_hits_ingested = False

        try:
            decision = self.task_classifier.classify(query)
            query_task = decision.task
            # Task lives on the user memory item + RunResult — avoid TASK_RECORD spam in OS.

            emb = self.llm.embed([query], input_type="query")[0]
            self.memory.upsert(
                MemoryItem(
                    text=f"USER_QUERY: {query}",
                    role_tag="user",
                    stage_tag=TaskStage.PLANNING.value,
                    source_agent="user",
                    session_id=session_id,
                    round=0,
                    doc_type=DocType.INTERACTION.value,
                    embedding=emb,
                    task=query_task,
                    metadata={
                        "task": query_task,
                        "task_confidence": decision.confidence,
                        "task_source": decision.source,
                        "task_rationale": decision.rationale,
                    },
                )
            )
            # Task classification is stored on the user memory item + RunResult.task
            # (not as separate TASK_RECORD docs — those bloated retrieval).

            routing_on = task_model_routing_enabled(task_model_routing)
            if routing_on:
                probe_result = probe_knowledge(
                    self.knowledge,
                    query,
                    embedding=emb,
                    query_task=query_task,
                )
                model_plan = build_run_model_plan(
                    query_task=query_task,
                    probe=probe_result,
                    registry=self.registry,
                    enabled=True,
                )
                # Reuse probe hits into working memory once (Searcher still may refresh).
                if probe_result.hits:
                    self._ingest_knowledge_hits(
                        probe_result.hits,
                        query=query,
                        session_id=session_id,
                        round_idx=0,
                        query_task=query_task,
                    )
                    probed_hits_ingested = True

            last_answer = ""
            for round_idx in range(1, max_rounds + 1):
                active_roles = self.activation.select(
                    query, round_idx=round_idx, prior_roles=prior_roles
                )
                for role in active_roles:
                    stage = STAGE_FOR_ROLE[role]
                    if role == AgentRole.SEARCHER:
                        if probed_hits_ingested and round_idx == 1:
                            # Already seeded from probe; skip duplicate first ingest.
                            probed_hits_ingested = False
                        else:
                            self._ingest_knowledge(
                                query, session_id, round_idx, emb, query_task=query_task
                            )

                    query_emb = emb
                    context = self.router.route(
                        query=query,
                        role=role,
                        session_id=session_id,
                        strategy=strategy_enum,
                        stage=stage,
                        embedding=query_emb,
                        query_task=query_task,
                    )
                    endpoint = self.registry.for_role(role)
                    planned = model_plan.for_role(role) if model_plan.enabled else None
                    if planned is not None:
                        endpoint = planned
                    resp = self.agents.run(
                        role=role,
                        query=query,
                        context_items=context,
                        usage=usage,
                        round_idx=round_idx,
                        model=model or endpoint.model,
                        api_base=endpoint.api_base,
                        api_key=endpoint.api_key,
                    )
                    out_emb = self.llm.embed([resp.text], input_type="passage")[0]
                    self.updater.update_from_agent_output(
                        text=resp.text,
                        role=role,
                        stage=stage,
                        session_id=session_id,
                        round_idx=round_idx,
                        doc_type=DocType.PLAN.value
                        if role == AgentRole.PLANNER
                        else DocType.FACT.value
                        if role == AgentRole.SEARCHER
                        else DocType.INTERACTION.value,
                        embedding=out_emb,
                        existing=self.memory.list_session(session_id),
                        task=query_task,
                    )
                    self.writeback.maybe_write(
                        text=resp.text,
                        role=role,
                        query=query,
                        session_id=session_id,
                        round_idx=round_idx,
                        embedding=out_emb,
                        task=query_task,
                    )
                    ctx_tokens = sum(i.token_length for i in context)
                    traces.append(
                        RoundTrace(
                            round_idx=round_idx,
                            role=role.value,
                            strategy=strategy_enum.value,
                            context_ids=[i.id for i in context],
                            context_texts=[i.text for i in context],
                            output=resp.text,
                            context_tokens=ctx_tokens,
                            model=resp.model,
                        )
                    )
                    prior_roles.append(role.value)
                    if role == AgentRole.RECOMMENDER:
                        last_answer = resp.text

            # Prefer recommender output; repair empty / citation-only finals.
            if is_citation_only(last_answer):
                for t in reversed(traces):
                    if t.role == "searcher" and not is_citation_only(t.output or ""):
                        last_answer = t.output
                        break
                if is_citation_only(last_answer):
                    for t in reversed(traces):
                        if not is_citation_only(t.output or ""):
                            last_answer = t.output
                            break

            # Compaction: never block the answer path by default.
            # background = schedule daemon thread after we build the result payload
            # sync = run inline (tests / RCR_COMPACT_AFTER_PROMPT=sync)
            mode = compact_mode()
            compaction: dict[str, Any] = {"status": "skipped", "deleted": 0}
            if mode == "sync":
                compaction = self.writeback.compact_after_prompt(
                    query,
                    embedding=emb,
                    task=query_task,
                ).as_dict()
                compaction["status"] = "sync"
            elif mode == "background" and self.writeback.writes:
                # Schedule after snapshot so the caller gets results first.
                compaction = {
                    "status": "pending_schedule",
                    "deleted": 0,
                    "kept_id": "",
                    "deleted_ids": [],
                }

            snapshot = [m.to_dict() for m in self.memory.list_session(session_id)]
            for row in snapshot:
                row.pop("embedding", None)

            elapsed = time.time() - started
            result = RunResult(
                session_id=session_id,
                answer=last_answer,
                strategy=strategy_enum.value,
                max_rounds=max_rounds,
                usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "calls": usage.calls,
                    "llm_traces": usage.traces,
                },
                traces=traces,
                memory_snapshot=snapshot,
                model_profile=self.registry.as_dict(),
                task=decision.as_dict(),
                knowledge_writebacks=list(self.writeback.writes),
                knowledge_compaction=compaction,
                budget=self.budget.as_dict(),
                token_backend=token_backend(),
                knowledge_probe=(probe_result.as_dict() if probe_result else {}),
                model_plan=model_plan.as_dict(),
                elapsed_sec=elapsed,
            )
            if compaction.get("status") == "pending_schedule":
                result.knowledge_compaction = schedule_compact_after_prompt(
                    self.writeback,
                    query,
                    embedding=emb,
                    task=query_task,
                )
            return result
        finally:
            self.writeback.enabled = prev_wb

    def _ingest_knowledge_hits(
        self,
        hits: list[MemoryItem],
        *,
        query: str,
        session_id: str,
        round_idx: int,
        query_task: str | None = None,
    ) -> None:
        for hit in hits:
            self.memory.upsert(
                MemoryItem(
                    text=hit.text,
                    id=f"know-{hit.id}-{session_id}-{round_idx}",
                    role_tag=AgentRole.SEARCHER.value,
                    stage_tag=TaskStage.SEARCHING.value,
                    source_agent="knowledge",
                    session_id=session_id,
                    round=round_idx,
                    doc_type=DocType.KNOWLEDGE.value,
                    embedding=hit.embedding,
                    score=hit.score,
                    token_length=count_tokens(hit.text or ""),
                    task=hit.task or query_task,
                    metadata={**(hit.metadata or {}), "query_task": query_task},
                )
            )

    def _ingest_knowledge(
        self,
        query: str,
        session_id: str,
        round_idx: int,
        embedding: list[float],
        *,
        query_task: str | None = None,
    ) -> None:
        hits = self.knowledge.search(
            query,
            k=12,
            embedding=embedding,
            task=query_task,
            soft_task=True,
        )
        hits = select_knowledge_for_context(hits, query=query, max_items=3)
        self._ingest_knowledge_hits(
            hits,
            query=query,
            session_id=session_id,
            round_idx=round_idx,
            query_task=query_task,
        )
