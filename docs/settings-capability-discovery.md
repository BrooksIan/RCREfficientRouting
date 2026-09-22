# Settings · capability discovery

The Streamlit **Settings** page lets operators register up to three OpenAI-compatible
LLM endpoints (model id, `/v1` base URL, API key). The app then **probes** each model
to estimate what it is good at and assigns agent roles accordingly.

## Flow

1. Enter 3 endpoints (Cloudera AI Inference / NIM, LiteLLM proxy aliases, etc.).
2. For each model, set an **intended task**: `generic`, `coding`, `reasoning`,
   `extraction`, `synthesis`, or `routing`.
3. Click **Discover capabilities**.
4. For each model we:
   - Apply a prior from the model id **and** the intended task
     (e.g. `coding` → planner/searcher bias, `extraction` → searcher).
   - Run four live probes: planner / searcher / recommender / router.
   - Score instruction-following and content quality with heuristics.
5. Greedy assignment picks the best remaining model per role (recommender → planner → searcher → router).
6. Assignments are stored in `.rcr/user_endpoints.json` and override the active `ModelRegistry`.

## Probe tasks

| Role | Probe asks the model to… |
| --- | --- |
| Planner | Emit exactly 3 search intents as a JSON array |
| Searcher | Extract factual bullets from a short passage |
| Recommender | Answer in one sentence from given facts |
| Router | Return a JSON list of agents to run |

## Storage

- Path: `.rcr/user_endpoints.json` (gitignored) or `RCR_USER_ENDPOINTS_PATH`
- Contains endpoints, capability scores, role assignments, timestamps
- API keys stay local — do not commit

## Using assignments

`ModelRegistry` loads discoveries when `RCR_USE_USER_ENDPOINTS=true` (default).
The Run page shows which discovered models are bound to Planner / Searcher / Recommender.

Dry-run without calling real endpoints: enable **Dry-run discovery (mock LLM)** on Settings.
