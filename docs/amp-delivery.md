# AMP delivery checklist

Use this before packaging or publishing **RCR — Role-Based Agent Routing** as a CML AMP.

## Pre-flight (secrets)

- [ ] Confirm `.env` and `.rcr/` are **not** in the publish tree (gitignored)
- [ ] Rotate any CDP / JWT tokens that were used in local testing
- [ ] Ship only `.env.example` and `deploy/env.cloudera.example` (placeholders)

## Catalog assets

- [x] `assets/images/cover.png` present
- [x] `community-amp-catalog-template.yaml` title/descriptions use **RCR**
- [x] `METADATA.yaml` / `.project-metadata.yaml` author & links filled
- [ ] Push cover to the public GitHub path referenced by `image_path`

## AMP structure

- [x] Numbered CML components: `0_` → `3_`
- [x] `cdsw-build.sh`, `LICENSE`, `README.md`
- [x] `.project-metadata.yaml` tasks: session → seed job → model → app

## Smoke (offline)

```bash
python -m pip install -e ".[dev]"
RCR_USE_OPENSEARCH=false RCR_USE_MOCK_LLM=true RCR_USE_USER_ENDPOINTS=false \
  python -m pytest -q --ignore=tests/test_phase5_hardening.py
```

Optional OpenSearch: start `deploy/docker-compose.opensearch.yaml`, set `RCR_USE_OPENSEARCH=true` and matching `OPENSEARCH_EMBED_DIM`.

## Operator notes for CML

1. Install AMP from Git / catalog.
2. Set project env: OpenSearch host (if used), LiteLLM / CAIIS bases, `CDP_TOKEN` or rely on `/tmp/jwt`.
3. Prefer `RCR_USE_MOCK_LLM=false` with real endpoints for demos; mock is fine for structure smoke.
4. Open the Streamlit app → Settings → register endpoints → Discover capabilities.
5. Run compare (`full` / `static` / `rcr`) with budget preset `savings`.
6. Leave **Route task to models** off unless ≥2 endpoints are configured and you want per-query rebinding.
