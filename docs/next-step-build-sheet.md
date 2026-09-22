# Next-step build sheet

Prioritized work after the §2 / architecture review
([`architecture.md`](architecture.md)). **Core Algorithm 1 is on track** — these
items close paper gaps and harden the AMP without rewriting the spine.

Source: [arXiv:2508.04903v3 §2](https://arxiv.org/html/2508.04903v3#Sx2).

## Priority order (do in sequence)

| P | Item | Why | Primary modules |
| --- | --- | --- | --- |
| **P1** | Scale budgets + real token accounting | ✅ Done — paper/demo presets + tiktoken accounting | `budget.py`, `tokens.py`, `router.py`, `metrics.py`, Streamlit |
| **P2** | Structured Memory Update | ✅ Done — YAML units, struct keys, conflict replace / plan prune | `memory_update.py`, `structured_memory.py`, WM schema |
| **P3** | LLM-judge Answer Quality | ✅ Done — optional judge behind flag; lexical default | `answer_judge.py`, `metrics.py`, Streamlit |
| **P4** | Real embeddings (CAIIS / LiteLLM) | ✅ Done — backend auto/mock/litellm, dim sync + recreate, seed/Settings probe | `embeddings.py`, `llm.py`, seed job, OpenSearch |
| **P5** | LiteLLM proxy in front of CAIIS | Aliases, retries, fewer CDP tokens in every process — better for CML AMP installs. | `deploy/docker-compose.litellm.yaml`, `.project-metadata.yaml` |
| **P6** | Tighten multi-agent loop | Force knowledge into final Recommender round; strip/limit reasoning noise so answers cite fixture facts reliably. | `orchestrator.py`, `agents.py`, budgets |
| **P7** | AMP / CML harden | Confirm env vars, shared OpenSearch seed, short operator runbook. | `.project-metadata.yaml`, `docs/` runbook |
| **P8** | Benchmark harness (optional) | ✅ Done (AMP fixtures) — HotPot/MuSiQue/2Wiki-*style* Full/Static/RCR run; ~70% context cut, F1 held. Not official dumps. | `scripts/run_multihop_experiment.py`, `docs/experiments/multihop-rcr-token-savings.md` |
| **P9** | Learned routing `π_route` (defer) | Paper allows heuristic *or* learned policy. Keep heuristic default; learned later. | `router.py` / training off-path |

## P1 — Budgets & token accounting (detail) ✅

- Default preset is **savings**: `β_base=256`, Planner/Searcher/Recommender → 768 / 512 / 640, full-context cap 4096 (keeps full baseline fat).
- Paper-scale: `RCR_BUDGET_PRESET=paper` (`β_base=512` → 2048 / 1536 / 1792).
- Fast local / unit demos: `RCR_BUDGET_PRESET=demo` (≈180 / 140 / 160).
- Optional fine-tune: `RCR_BUDGET_SCALE=0.75` multiplies resolved budgets.
- `MemoryItem.token_length` via `tokens.count_tokens` (tiktoken → LiteLLM → char/4 fallback).
- UI / metrics: routed **context tokens** (tokenizer) vs LLM **prompt/completion** usage, plus budget caption.

**Done when:** Same fixture query shows RCR token totals from usage (or a real tokenizer), and absolute budgets are no longer demo-toy (~100s of words) when measuring savings.

## P2 — Structured Memory Update (detail) ✅

- Persist structured fields (`struct_key`, `struct_kind`, `structure`) plus YAML `text` for BM25.
- Kinds: `plan_step`, `fact`, `evidence`, `answer`.
- Conflict / replace: same `struct_key` reuses id (overwrite); shorter revised plans prune leftover `plan_step:*`.
- Novelty filter kept; compaction unchanged (structure sits above it).
- Query helpers: `MemoryUpdater.get_by_key`, `list_by_kind`.

**Done when:** Working memory items after a round are queryable by role/stage *and* by structured keys; duplicate contradictory facts are resolved rather than stacked.

## P3 — Answer Quality Score (detail) ✅

- Optional LLM-as-judge (1–5, paper Appendix A.4 prompt) behind `RCR_LLM_JUDGE=1` or the Run-page checkbox.
- Lexical proxy remains the default offline / smoke path (`answer_quality` / `lexical_quality`).
- When enabled, metrics include `judge_quality` + `judge_justification`; Streamlit compare shows both.

**Done when:** Compare lab can report lexical + optional judge score without slowing default smoke path.

## P4 — Real embeddings (detail) ✅

- `RCR_EMBED_BACKEND=auto|mock|litellm` — auto uses live embeds when embed model + base/key exist.
- `prepare_embeddings` / Settings **Probe** report backend + dim; **Sync indices** recreates knn mappings on mismatch (`RCR_RECREATE_INDICES_ON_DIM_MISMATCH`).
- Seed job probes, syncs dim, stamps `embed_backend` / `embed_dim` on seeded docs.
- Chat-only CAIIS (e.g. Nemotron) still falls back to mock unless a dedicated embed endpoint is set.

**Done when:** With a real embed endpoint configured, probe reports `backend=litellm` and OpenSearch knn dim matches; mock remains the offline default.

## Product vs paper (keep labeled)

Do **not** block the sheet on these; they are AMP extensions already in place:

- LiteLLM / Cloudera multi-model + Settings capability discovery
- Prompt task classification (`w_task` in `α`)
- Knowledge write-back + post-prompt compaction
- Optional LLM agent-activation router (`RCR_LLM_AGENT_ROUTER`)

## Explicit non-goals (still)

- Full paper-scale HotPotQA / MuSiQue / 2Wiki as a gate for the AMP
- Mandatory learned `π_route` before shipping
- Equating LiteLLM RBAC with Planner / Searcher / Recommender

## Status snapshot

| Area | Status |
| --- | --- |
| Algorithm 1 (score → sort → greedy fill) | On track |
| Shared memory + knowledge (OpenSearch / in-memory) | On track |
| Full / Static / RCR baselines | On track |
| P1–P3 paper alignment | **P1–P3 done** |
| P4–P7 AMP hardening | **P4 done**; P5–P7 next |
| P8–P9 | **P8 AMP multi-hop experiment done**; P9 defer |
