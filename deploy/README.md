# Deploy

## Local OpenSearch (OSS)

```bash
docker compose -f deploy/docker-compose.opensearch.yaml up -d
# optional Dashboards:
docker compose -f deploy/docker-compose.opensearch.yaml --profile ui up -d
```

Environment for the app/jobs:

```bash
export RCR_USE_OPENSEARCH=true
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
export OPENSEARCH_EMBED_DIM=8   # match MockLLM; set to real model dim in production
export RCR_USE_MOCK_LLM=true    # or provide OPENAI_API_KEY / LITELLM_* and set false
```

Index mappings: [`opensearch_index_templates.json`](opensearch_index_templates.json)

## LiteLLM + Cloudera AI Inference

See [`docs/cloudera-litellm.md`](../docs/cloudera-litellm.md) for the full guide.

Quick start (proxy in front of CAIIS):

```bash
cp deploy/env.cloudera.example .env
# edit CLOUDERA_* and CDP_TOKEN

python scripts/render_litellm_cloudera_config.py \
  --out deploy/litellm_config.cloudera.generated.yaml

docker compose -f deploy/docker-compose.litellm.yaml --env-file .env up -d
python scripts/check_cloudera_llm.py
```

Point the app at the proxy:

```bash
export RCR_USE_MOCK_LLM=false
export LITELLM_API_BASE=http://localhost:4000
export LITELLM_API_KEY=sk-rcr-local
export LITELLM_MODEL=cloudera-chat
export LITELLM_EMBED_MODEL=cloudera-embed
```

Per-role model overrides (optional):

```bash
export LITELLM_MODEL=cloudera-chat
export LITELLM_MODEL_PLANNER=cloudera-chat
export LITELLM_MODEL_SEARCHER=cloudera-chat
export LITELLM_MODEL_RECOMMENDER=cloudera-chat-alt
```
