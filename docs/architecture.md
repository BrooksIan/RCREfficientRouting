# Architecture: Role-Based Agent Routing (RCR + OpenSearch + LiteLLM)

This blueprint reimplements the **RCR-Router** approach from
[arXiv:2508.04903v3 §2](https://arxiv.org/html/2508.04903v3#Sx2)
(*RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems
with Structured Memory*).

## Verdict

**Core Algorithm 1 is on track.** Planner / Searcher / Recommender, shared memory
`M_t`, role+stage-conditioned importance scoring, role budgets `B_i`, and greedy
fill under the budget match the paper. Cloudera AMP pieces (LiteLLM, Settings
discovery, task labels, knowledge write-back) are **product extensions** layered
on that core — keep them, but don’t confuse them with §2 routing itself.

## Layer boundaries

| Concern | Owner |
| --- | --- |
| Persist / retrieve shared memory and knowledge | **OpenSearch OSS** (BM25 + k-NN hybrid) |
| Which memory subset each agent sees under a token budget | **RCR-Router** (`src/rcr_router`) — Algorithm 1 |
| LLM completions and embeddings | **LiteLLM** (`litellm.completion` / `litellm.embedding`) |
| Key/team model access control | LiteLLM RBAC (optional proxy; not agent roles) |

Do **not** equate LiteLLM RBAC roles (`proxy_admin`, `internal_user`, …) with
agent roles (`Planner`, `Searcher`, `Recommender`).

## Paper §2 → modules

| Paper concept | Module |
| --- | --- |
| Shared Memory Store `M_t` | OpenSearch `rcr-working-memory` via `opensearch_store.py` (or in-memory fallback) |
| Task-relevant knowledge / corpus | OpenSearch `rcr-knowledge` |
| Token Budget Allocator `B_i = β_base + β_role(R_i)` | `budget.py` |
| Importance Scorer `α(m; R_i, S_t)` | `scorer.py` (retrieval + embedding + role + stage + recency; optional `task`) |
| Semantic Filter + Routing (Algorithm 1) | `router.py` — score → sort → near-dedupe → greedy fill |
| Memory Update | `memory_update.py` + `structured_memory.py` → YAML units, `struct_key` conflict resolve |
| Iterative routing `t=1..T` | `orchestrator.py` (default `T=3`) |
| Agent LLM queries | `agents.py` → `llm.py` |
| Full-Context / Static / RCR baselines | `RoutingStrategy` in `models.py` + Streamlit compare |
| Answer Quality Score (paper metric) | Lexical proxy + optional LLM judge (`metrics.py`, `answer_judge.py`) |

### Algorithm 1 loop (faithful)

```text
for t in 1..T:
  for agent role R_i in {Planner, Searcher, Recommender}:
    score each m ∈ M_t with α(m; R_i, S_t)
    sort by α descending
    greedily fill C_t^i until Σ TokenLength(m) ≤ B_i
    LLM_output = LLM(Prompt(C_t^i))
    M ← Update(M, structured agent output)
```

## Indices

- `rcr-working-memory` — session/round-scoped agent outputs, plans, facts (`M_t`)
- `rcr-knowledge` — seeded corpus + optional canonical write-backs (`learned-q-*`)

Embeddings via LiteLLM when configured (`RCR_EMBED_BACKEND=auto`); mock/hash when
`RCR_EMBED_BACKEND=mock` or the remote `/embeddings` call fails (optional fallback).
OpenSearch knn dim is probed and synced (`prepare_embeddings`); recreate indices on
mismatch via `RCR_RECREATE_INDICES_ON_DIM_MISMATCH` or Settings.

## Product extensions (beyond §2)

| Extension | Purpose |
| --- | --- |
| LiteLLM + Cloudera endpoints | Hosted model access |
| Settings capability discovery | Assign best LLM per agent role |
| Prompt task classification | Soft-boost retrieval / `α` by query task |
| Knowledge write-back + optional compact-after-prompt | Persist novel answers; default **background** compact after return |

| Optional LLM agent activation router | Dynamic which agents run (`RCR_LLM_AGENT_ROUTER`) |
| Exact-text dedupe in `route()` | Light guard; near-dupe opt-in via `RCR_ROUTE_NEAR_DEDUPE` |

## Known gaps vs the paper (honest)

| Paper | Current project | Suggested direction |
| --- | --- | --- |
| Budgets often ~512–4096 tokens in experiments | Default **savings** preset (Planner/Searcher/Recommender → 768 / 512 / 640, full cap 4096) for clear RCR cuts; **paper** (~512–2k `B_i`) and **demo** (tiny) also available; tiktoken/LiteLLM counts | `RCR_BUDGET_PRESET=paper\|savings\|demo`, `RCR_BUDGET_BASE`, `RCR_BUDGET_SCALE` |
| Memory Update: extract → filter → **structure** (YAML/graphs) → conflict resolve | Extract → novelty filter → YAML units with `struct_key` / `struct_kind` → same-key replace + plan supersession | Extend to graph triples later if needed |
| Heuristic **or learned** routing policy | Heuristic only | Keep heuristic as default; learned π later |
| Answer Quality Score (LLM-as-judge style) | Lexical proxy always; optional LLM judge via `RCR_LLM_JUDGE` / UI checkbox (`answer_judge.py`) | Keep lexical default for smoke; enable judge for paper-style compare |
| HotPotQA / MuSiQue / 2Wiki eval | Demo corpus + Streamlit compare | Add benchmark harness if claiming paper-parity numbers |

Prioritized follow-ups live in
[`next-step-build-sheet.md`](next-step-build-sheet.md) (P1 budgets → P2 structured
memory update → P3 LLM-judge quality, then AMP hardening).

## Routing strategies (demo baselines)

- **full** — large recall from working memory (practical cap + near-dedupe)
- **static** — role/stage tag filters only
- **rcr** — hybrid retrieval + `α` + greedy token budget (Algorithm 1)

## Citation

Liu et al., *RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent
LLM Systems with Structured Memory*, arXiv:2508.04903v3, 2025.
