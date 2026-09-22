# Prompt task classification → knowledge

When a user prompt arrives, the orchestrator **classifies a task**, persists it, and
uses it so agents retrieve task-relevant knowledge and context.

## Shared taxonomy

Same labels as Settings model `intended_task`:

`generic` · `coding` · `reasoning` · `extraction` · `synthesis` · `routing`

| Concept | Meaning |
| --- | --- |
| Model `intended_task` | What you bought the endpoint for |
| Query `task` | What **this prompt** needs |

## Flow

1. `TaskClassifier` runs at the start of `Orchestrator.run`.
2. Default mode `RCR_TASK_CLASSIFIER=heuristic` (keyword priors). Set `llm` or `hybrid` to refine with the chat model.
3. The user query is written to working memory with `task` + metadata (`task_confidence`, `task_source`).
4. A durable **TaskRecord** is upserted into the knowledge index (`doc_type=knowledge`, `task=…`).
5. Searcher knowledge search soft-boosts same-task docs (hard filter available via `soft_task=False`).
6. RCR `ImportanceScorer` adds a task-match term (`w_task≈0.14`).
7. Agent outputs inherit the query `task` when upserted to working memory.
8. **Knowledge write-back:** At most **one canonical doc per query** (`learned-q-*`).
   Near-duplicate paraphrases are skipped; Searcher/Recommender competition keeps the
   better (usually shorter ANSWER) document.
9. Retrieval / RCR use **exact-text** dedupe by default (token budget still caps
   context). Near-dupe in the hot path is opt-in via `RCR_ROUTE_NEAR_DEDUPE` — it was
   O(n²) and could drop complementary facts.
10. `TASK_RECORD` docs are no longer written to the knowledge index (task lives on the
    user memory item + `RunResult.task`).
11. Post-prompt compaction default is **`background`**: answer returns first, then a
    daemon thread runs `compact_after_prompt` for that query. Use `sync` to block the
    run, or `off` to disable. Manual / Settings sweep still force-compacts.
12. Manual / Settings sweep: **Compact knowledge now** on the Settings page, or
    `python scripts/compact_knowledge.py` (both force compaction).

## Env

| Variable | Default | Notes |
| --- | --- | --- |
| `RCR_TASK_CLASSIFIER` | `heuristic` | `heuristic` \| `llm` \| `hybrid` \| `off` |
| `RCR_KNOWLEDGE_WRITEBACK` | `true` | Persist novel Searcher/Recommender findings to knowledge |
| `RCR_COMPACT_AFTER_PROMPT` | `background` | `background` (after return) \| `sync` \| `off` |
| `RCR_ROUTE_NEAR_DEDUPE` | `false` | Near-dupe in `route()`; leave off unless corpus is very spammy |

## OpenSearch

Index mapping includes `task` (keyword) and `metadata.task*`. Recreate indices or update the template after pull if upgrading an existing cluster (`deploy/opensearch_index_templates.json`).
