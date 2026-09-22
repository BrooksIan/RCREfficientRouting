from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "compare_strategies", ROOT / "scripts" / "compare_strategies.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
compare = _mod.compare
FIXTURE_QUERIES = _mod.FIXTURE_QUERIES


def test_comparison_covers_strategies_and_queries():
    rows = compare(max_rounds=1)
    assert len(rows) == len(FIXTURE_QUERIES) * 3
    strategies = {r["strategy"] for r in rows}
    assert strategies == {"full", "static", "rcr"}
    for q in FIXTURE_QUERIES:
        subset = [r for r in rows if r["query"] == q]
        assert {r["strategy"] for r in subset} == {"full", "static", "rcr"}
        by_s = {r["strategy"]: r["context_tokens"] for r in subset}
        # With near-dupe compaction, full may not always exceed RCR on tokens;
        # still expect all strategies to route some context and stay in quality band.
        assert by_s["full"] > 0 and by_s["rcr"] > 0 and by_s["static"] >= 0
        assert all(1.0 <= r["answer_quality"] <= 5.0 for r in subset)


def test_model_api_wrapper():
    import importlib.machinery

    path = Path(__file__).resolve().parents[1] / "2_model-deploy-router" / "launch-model.py"
    loader = importlib.machinery.SourceFileLoader("launch_model", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    out = mod.run_query(
        {"prompt": "What drove Acme Corp revenue growth in Q3?", "strategy": "rcr", "max_rounds": 1}
    )
    assert "response" in out
    assert out["response"]["answer"]
    assert out["response"]["strategy"] == "rcr"
    assert out["response"]["traces"]
