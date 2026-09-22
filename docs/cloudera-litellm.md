# Cloudera-hosted LLMs via LiteLLM

This AMP calls models through **LiteLLM**, pointed at **Cloudera AI Inference Service (CAIIS)** OpenAI-compatible endpoints (NVIDIA NIM).

Official CAIIS OpenAI client notes:
https://docs.cloudera.com/machine-learning/cloud/ai-inference/topics/ml-caii-make-inference-call-model-endpoint-with-openai-api.html

LiteLLM OpenAI-compatible provider:
https://docs.litellm.ai/docs/providers/openai_compatible

## Recommended topology

```text
RCR agents  →  LiteLLM proxy (:4000)  →  Cloudera AI Inference (/v1/chat/completions)
                     │
                     └──→ Cloudera embed endpoint (/v1/embeddings)  [optional]
```

Running a LiteLLM proxy keeps CDP tokens and endpoint URLs out of the Streamlit/model process and enables retries, aliases, and per-role model pins.

## 1. Collect endpoint details (CAIIS UI)

From **Model Endpoint Details**:

1. Copy the endpoint URL and trim to `.../v1` (remove trailing path segments such as `/chat/completions`).
2. Note the **MODEL_NAME** from the Cloudera AI Registry registration.
3. Obtain a workload auth token:
   - Local / automation: `export CDP_TOKEN=...`
   - CML Workbench session: token is often available at `/tmp/jwt` (`access_token`); this project reads it automatically via `rcr_router.cloudera.resolve_api_key()`.

## 2. Start the LiteLLM proxy

```bash
cp deploy/env.cloudera.example .env
# edit .env with real CLOUDERA_* and CDP_TOKEN values

docker compose -f deploy/docker-compose.litellm.yaml --env-file .env up -d
curl -s http://localhost:4000/health/liveliness
```

Config file: [`deploy/litellm_config.cloudera.yaml`](../deploy/litellm_config.cloudera.yaml)

Aliases exposed by the proxy:

| Alias | Purpose |
| --- | --- |
| `cloudera-chat` | Default chat / completion model |
| `cloudera-embed` | Embedding model (or same chat endpoint if none) |
| `cloudera-chat-alt` | Optional second deployment for failover / recommender |

## 3. Point the AMP / local app at the proxy

```bash
export RCR_USE_MOCK_LLM=false
export LITELLM_API_BASE=http://localhost:4000
export LITELLM_API_KEY=sk-rcr-local          # LITELLM_MASTER_KEY
export LITELLM_MODEL=cloudera-chat
export LITELLM_EMBED_MODEL=cloudera-embed
export RCR_EMBED_BACKEND=auto               # or litellm; mock for offline
export OPENSEARCH_EMBED_DIM=<real-dim>      # auto-updated on successful probe
# After changing embed model/dim:
export RCR_RECREATE_INDICES_ON_DIM_MISMATCH=true
python 1_job-seed-role-catalog/job.py
```

Connectivity check:

```bash
python scripts/check_cloudera_llm.py
```

Smoke with real LLM:

```bash
python -m rcr_router.smoke
# or
python 2_model-deploy-router/launch-model.py
```

## Embeddings without a CAIIS embed endpoint

Nemotron (and most chat-only NIM endpoints) do **not** implement `/v1/embeddings`.
For local / demo semantic search:

```bash
export RCR_EMBED_BACKEND=local          # hashed n-gram vectors (dim 384)
export OPENSEARCH_EMBED_DIM=384
export RCR_RECREATE_INDICES_ON_DIM_MISMATCH=true
python 1_job-seed-role-catalog/job.py
```

Or install optional `fastembed` and keep `RCR_EMBED_BACKEND=local` for ONNX `bge-small`.

When you have a dedicated embed endpoint:

```bash
export CLOUDERA_AI_INFERENCE_EMBED_API_BASE=https://.../endpoints/<embed-name>/v1
export CLOUDERA_AI_INFERENCE_EMBED_MODEL=<embed-model-id>
export RCR_EMBED_BACKEND=auto
```

For a CML session that already has network access to CAIIS:

```bash
export RCR_USE_MOCK_LLM=false
export RCR_FORCE_DIRECT_CAIIS=true
export CLOUDERA_AI_INFERENCE_API_BASE=https://.../endpoints/<name>/v1
export CLOUDERA_AI_INFERENCE_MODEL=<registry-model-name>
export CDP_TOKEN=...   # or rely on /tmp/jwt inside CML
```

LiteLLM will call `openai/<MODEL>` with that `api_base` / `api_key`.

## 5. CML AMP environment variables

Set these in the AMP install / project settings (also declared in `.project-metadata.yaml`):

- `RCR_USE_MOCK_LLM=false`
- `LITELLM_API_BASE` (proxy URL reachable from CML) **or** direct `CLOUDERA_AI_INFERENCE_API_BASE`
- `LITELLM_MODEL` / `LITELLM_EMBED_MODEL`
- `CDP_TOKEN` or `CLOUDERA_AI_INFERENCE_API_KEY`
- `OPENSEARCH_EMBED_DIM` matching the embed model

## Troubleshooting

| Symptom | Likely fix |
| --- | --- |
| `Not Found` from LiteLLM | Ensure `api_base` ends with `/v1` |
| 401 / 403 | Refresh `CDP_TOKEN` / CML JWT |
| Embeddings fail | Set a dedicated embed endpoint (`CLOUDERA_AI_INFERENCE_EMBED_*`) — chat-only models often lack `/embeddings`. Or `RCR_EMBED_BACKEND=mock` for offline. |
| Wrong vector dim in OpenSearch | Settings → **Sync indices to embed dim**, then **Re-seed knowledge**. Or `RCR_RECREATE_INDICES_ON_DIM_MISMATCH=true` + seed job. |
| Probe shows `backend=mock` with CAIIS set | Point embed vars at an **embedding** model, not Nemotron chat; set `RCR_EMBED_BACKEND=auto` (not `mock`). |
