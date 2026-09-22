#!/usr/bin/env python3
"""Render a concrete LiteLLM config for Cloudera AI Inference from env vars.

Supports a shared default endpoint plus optional per-role overrides:
  CLOUDERA_AI_INFERENCE_API_BASE_PLANNER / _MODEL_PLANNER
  CLOUDERA_AI_INFERENCE_API_BASE_SEARCHER / _MODEL_SEARCHER
  ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _req(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise SystemExit(f"Missing required env var: {name}")
    return val


def _opt(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip() or default


def openai_model(name: str) -> str:
    return name if "/" in name else f"openai/{name}"


def _entry(alias: str, model: str, api_base: str, api_key_env: str) -> dict:
    return {
        "model_name": alias,
        "litellm_params": {
            "model": openai_model(model),
            "api_base": api_base.rstrip("/"),
            "api_key": f"os.environ/{api_key_env}",
            "drop_params": True,
        },
    }


def build_config() -> dict:
    api_base = _req("CLOUDERA_AI_INFERENCE_API_BASE").rstrip("/")
    chat_model = _req("CLOUDERA_AI_INFERENCE_MODEL")
    api_key_env = "CDP_TOKEN" if os.getenv("CDP_TOKEN") else "CLOUDERA_AI_INFERENCE_API_KEY"
    if not os.getenv(api_key_env):
        raise SystemExit("Set CDP_TOKEN or CLOUDERA_AI_INFERENCE_API_KEY")

    embed_base = _opt("CLOUDERA_AI_INFERENCE_EMBED_API_BASE", api_base).rstrip("/")
    embed_model = _opt("CLOUDERA_AI_INFERENCE_EMBED_MODEL", chat_model)
    master = _opt("LITELLM_MASTER_KEY", "sk-rcr-local")

    model_list = [
        _entry("cloudera-chat", chat_model, api_base, api_key_env),
        _entry("cloudera-embed", embed_model, embed_base, api_key_env),
    ]

    # Role-specific aliases for split-via-proxy profile
    for role in ("planner", "searcher", "recommender", "router"):
        role_u = role.upper()
        role_model = _opt(f"CLOUDERA_AI_INFERENCE_MODEL_{role_u}", chat_model)
        role_base = _opt(f"CLOUDERA_AI_INFERENCE_API_BASE_{role_u}", api_base)
        model_list.append(_entry(f"cloudera-{role}", role_model, role_base, api_key_env))

    # Back-compat alt alias
    alt_model = _opt("CLOUDERA_AI_INFERENCE_MODEL_ALT", chat_model)
    model_list.append(_entry("cloudera-chat-alt", alt_model, api_base, api_key_env))

    return {
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": "simple-shuffle",
            "num_retries": 2,
            "timeout": 120,
        },
        "litellm_settings": {"drop_params": True, "set_verbose": False},
        "general_settings": {"master_key": master},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "deploy" / "litellm_config.cloudera.generated.yaml",
    )
    args = parser.parse_args()
    cfg = build_config()
    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.out} ({len(cfg['model_list'])} models)")


if __name__ == "__main__":
    main()
