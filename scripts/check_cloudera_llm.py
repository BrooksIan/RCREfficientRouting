#!/usr/bin/env python3
"""Verify LiteLLM can reach Cloudera-hosted (or proxy) chat/embed endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcr_router.cloudera import resolve_cloudera_settings
from rcr_router.llm import LiteLLMClient, build_llm_client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Reply with the single word: pong")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    # Force non-mock for this check
    os.environ["RCR_USE_MOCK_LLM"] = "false"

    settings = resolve_cloudera_settings()
    if settings is None:
        print("No Cloudera/LiteLLM settings resolved.")
        print("Set LITELLM_API_BASE (+ LITELLM_API_KEY) or CLOUDERA_AI_INFERENCE_API_BASE (+ CDP_TOKEN).")
        print("See docs/cloudera-litellm.md")
        return 2

    print("Resolved settings:")
    print(
        json.dumps(
            {
                "api_base": settings.api_base,
                "via_proxy": settings.via_proxy,
                "chat_model": settings.litellm_chat_model,
                "embed_model": settings.litellm_embed_model,
                "api_key_set": bool(settings.api_key),
            },
            indent=2,
        )
    )

    client = build_llm_client(force_mock=False)
    if not isinstance(client, LiteLLMClient):
        print(f"Unexpected client type: {type(client)}")
        return 1

    print("\nChat completion…")
    resp = client.complete(
        [
            {"role": "system", "content": "You are a connectivity probe."},
            {"role": "user", "content": args.prompt},
        ]
    )
    print(f"model={resp.model} tokens={resp.prompt_tokens}+{resp.completion_tokens}")
    print(f"text={resp.text!r}")

    if not args.skip_embed:
        print("\nEmbedding…")
        # Prefer live path for this connectivity check
        os.environ.setdefault("RCR_EMBED_BACKEND", "litellm")
        os.environ["RCR_EMBED_FALLBACK_MOCK"] = "false"
        from rcr_router.embeddings import probe_embeddings

        try:
            status = probe_embeddings(client)
            print(
                f"backend={status.backend} dim={status.dim} "
                f"model={status.model or 'n/a'} ok={status.ok}"
            )
            if not status.ok or status.backend != "litellm":
                print(
                    "Live embeddings not confirmed. Configure "
                    "CLOUDERA_AI_INFERENCE_EMBED_* (dedicated embed endpoint) "
                    "or LITELLM_EMBED_MODEL via proxy. Chat-only models often lack /embeddings."
                )
                if status.detail:
                    print(f"detail={status.detail}")
                return 0
        except Exception as exc:  # noqa: BLE001
            print(f"embed failed: {exc}")
            print("Chat OK; configure CLOUDERA_AI_INFERENCE_EMBED_* if embeddings are required.")
            return 0

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
