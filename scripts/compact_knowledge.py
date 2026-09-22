#!/usr/bin/env python3
"""Manual full-index compaction (optional).

Per-prompt compaction is **off by default** (`RCR_COMPACT_AFTER_PROMPT=false`).
Use this script or the Settings → **Compact knowledge now** button to force a sweep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass

from rcr_router.factory import build_orchestrator  # noqa: E402
from rcr_router.knowledge_writeback import KnowledgeWriteback  # noqa: E402


def main() -> None:
    use_os = True
    orch = build_orchestrator(use_opensearch=use_os, force_mock_llm=True)
    result = KnowledgeWriteback(orch.knowledge).compact_corpus()
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
