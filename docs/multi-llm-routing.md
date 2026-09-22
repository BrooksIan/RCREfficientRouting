# Multi-LLM role routing

RCR context routing (what memory each agent sees) is separate from **model routing**
(which LLM serves each agent). LiteLLM is the single client for all model calls.

## Why multiple LLMs

| Role | Typical model traits |
| --- | --- |
| **Router** (optional) | Fast/cheap; decides which agents run this round |
| **Planner** | Strong reasoning / decomposition |
| **Searcher** | Faithful extraction; can be smaller |
| **Recommender** | High-quality synthesis / final answer |

Today one Cloudera endpoint (`nvidia/nemotron-3-super-120b-a12b`) can back every role.
When you add more CAIIS endpoints, assign them per role without changing agent code.

## Profiles

Configured in [`assets/config/model_profiles.yaml`](../assets/config/model_profiles.yaml):

| Profile | Use when |
| --- | --- |
| `single-nemotron` | One shared chat model (default) |
| `split-via-proxy` | LiteLLM aliases `cloudera-planner` / `-searcher` / `-recommender` / `-router` |
| `split-direct` | Different CAIIS `api_base` per role via env |

```bash
export RCR_MODEL_PROFILE=split-via-proxy
# or
export RCR_MODEL_PROFILE=split-direct
export CLOUDERA_AI_INFERENCE_API_BASE_PLANNER=https://.../endpoints/planner/v1
export CLOUDERA_AI_INFERENCE_MODEL_PLANNER=nvidia/nemotron-3-super-120b-a12b
export CLOUDERA_AI_INFERENCE_API_BASE_SEARCHER=https://.../endpoints/searcher/v1
export CLOUDERA_AI_INFERENCE_MODEL_SEARCHER=some-smaller-model
export CLOUDERA_AI_INFERENCE_API_BASE_RECOMMENDER=https://.../endpoints/recommender/v1
export CLOUDERA_AI_INFERENCE_MODEL_RECOMMENDER=nvidia/nemotron-3-super-120b-a12b
```

Env overrides always win over the YAML profile.

## Optional LLM agent activation

```bash
export RCR_LLM_AGENT_ROUTER=true
```

A dedicated **router** model returns which of `planner` / `searcher` / `recommender`
should run each round (JSON list). Default remains fixed order when disabled.

## Render LiteLLM proxy with role aliases

```bash
python scripts/render_litellm_cloudera_config.py \
  --out deploy/litellm_config.cloudera.generated.yaml
```

Produces aliases: `cloudera-chat`, `cloudera-embed`, `cloudera-planner`,
`cloudera-searcher`, `cloudera-recommender`, `cloudera-router`, `cloudera-chat-alt`.

Then:

```bash
export RCR_MODEL_PROFILE=split-via-proxy
export LITELLM_API_BASE=http://localhost:4000
export LITELLM_API_KEY=sk-rcr-local
export RCR_USE_MOCK_LLM=false
export RCR_FORCE_DIRECT_CAIIS=false
```

## Inspect

Traces include `model` per agent turn. Streamlit shows the active profile and
per-role model labels. Programmatically: `result.model_profile`.
