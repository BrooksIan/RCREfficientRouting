#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
PORT="${CDSW_APP_PORT:-${PORT:-8501}}"
exec streamlit run "${ROOT}/3_app-run-routing-ui/app.py" \
  --server.port "${PORT}" \
  --server.address 127.0.0.1
