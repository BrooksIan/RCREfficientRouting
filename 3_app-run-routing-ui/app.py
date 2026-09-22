"""Streamlit app: Run workspace + Settings for 3 LLM endpoints with capability discovery."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load local .env and OVERRIDE process env. CML AMP / project-metadata often
# injects RCR_USE_MOCK_LLM=true as a default; without override=True the UI stays
# on MockLLMClient (probe shows backend=mock model=n/a) even when .env has
# real CAIIS embed settings.
def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=True)
    except Exception:
        # Minimal fallback if python-dotenv is missing
        env_path = ROOT / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'").strip('"')
            if k:
                os.environ[k] = v


_load_project_env()

# Streamlit keeps imported packages across script reruns; always take a fresh
# copy of our package so Settings/Run pick up endpoint_store + orchestrator edits.
for _name in list(sys.modules):
    if _name == "rcr_router" or _name.startswith("rcr_router."):
        sys.modules.pop(_name, None)

import streamlit as st

from rcr_router.capability_probe import run_discovery
from rcr_router.embeddings import prepare_embeddings, probe_embeddings
from rcr_router.endpoint_store import (
    UserEndpoint,
    blank_endpoints,
    default_store_path,
    INTENDED_TASK_LABELS,
    INTENDED_TASKS,
    load_store,
    save_store,
)
from rcr_router.factory import build_orchestrator
from rcr_router.knowledge_writeback import KnowledgeWriteback
from rcr_router.metrics import summarize_run
from rcr_router.models import DocType, MemoryItem, RoutingStrategy
from rcr_router.opensearch_store import (
    clear_working_memory,
    opensearch_available,
    working_memory_count,
)


def seed_knowledge(orch) -> None:
    corpus_path = ROOT / "assets" / "data" / "knowledge-corpus.json"
    rows = json.loads(corpus_path.read_text(encoding="utf-8"))
    embs = orch.llm.embed([r["text"] for r in rows], input_type="passage")
    for row, emb in zip(rows, embs):
        orch.knowledge.upsert(
            MemoryItem(
                id=row["id"],
                text=row["text"],
                doc_type=row.get("doc_type", DocType.KNOWLEDGE.value),
                embedding=emb,
            )
        )


def _opensearch_enabled() -> bool:
    flag = os.getenv("RCR_USE_OPENSEARCH", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    # Auto: use OpenSearch when a host is configured (local compose default).
    return bool(os.getenv("OPENSEARCH_HOST") or os.getenv("OPENSEARCH_PORT"))


def get_orchestrator(force_reload: bool = False):
    """Build orchestrator; bust cache when settings change."""
    _load_project_env()
    cache_key = "orch"
    if force_reload and cache_key in st.session_state:
        del st.session_state[cache_key]
    if cache_key not in st.session_state:
        use_os = _opensearch_enabled()
        orch = build_orchestrator(use_opensearch=use_os)
        seed_knowledge(orch)
        st.session_state[cache_key] = orch
        st.session_state["knowledge_backend"] = (
            "opensearch" if use_os else "in-memory"
        )
        st.session_state["llm_client_type"] = type(orch.llm).__name__
        st.session_state["embed_model"] = getattr(orch.llm, "embed_model", None) or ""
    return st.session_state[cache_key]


def prune_working_memory_if_needed(*, force: bool = False) -> int:
    """Drop accumulated working-memory docs so OpenSearch search stays fast."""
    if not _opensearch_enabled() or not opensearch_available():
        return 0
    count = working_memory_count()
    threshold = int(os.getenv("RCR_WM_PRUNE_THRESHOLD", "80"))
    if force or count >= threshold:
        deleted = clear_working_memory()
        st.session_state["last_wm_prune"] = {
            "before": count,
            "deleted": deleted,
            "threshold": threshold,
        }
        return deleted
    return 0


def wm_prune_threshold() -> int:
    return int(os.getenv("RCR_WM_PRUNE_THRESHOLD", "80"))


def render_result(result, query: str, *, use_judge: bool = False, llm=None) -> None:
    metrics = summarize_run(result, query, llm=llm, use_judge=use_judge)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Prompt tokens", metrics.prompt_tokens)
    m2.metric("Completion tokens", metrics.completion_tokens)
    m3.metric("Context tokens", metrics.context_tokens)
    m4.metric("LLM calls", metrics.llm_calls)
    m5.metric("Elapsed (s)", f"{metrics.elapsed_sec:.2f}")
    m6.metric("Lexical quality", metrics.lexical_quality or metrics.answer_quality)

    if metrics.judge_enabled and metrics.judge_quality is not None:
        j1, j2 = st.columns(2)
        j1.metric("Judge quality", metrics.judge_quality)
        j2.caption(metrics.judge_justification or "(no justification)")
    elif use_judge and metrics.judge_quality is None:
        st.caption("LLM judge enabled but no score returned — see notes below.")

    budget = getattr(result, "budget", None) or {}
    backend = getattr(result, "token_backend", "") or metrics.token_backend
    if budget or backend:
        roles = budget.get("roles") or {}
        role_bits = ", ".join(f"{k}={v}" for k, v in roles.items()) if roles else ""
        st.caption(
            f"Routed **context tokens** use tokenizer `{backend or 'n/a'}` "
            f"(preset `{budget.get('preset') or metrics.budget_preset or 'n/a'}`, "
            f"B_i: {role_bits or 'n/a'}). "
            "Prompt/completion come from LLM usage — compare them separately."
        )

    model_profile = getattr(result, "model_profile", None) or {}
    if model_profile:
        st.caption(
            f"Model profile: **{model_profile.get('profile')}** — "
            + ", ".join(
                f"{role}→`{(info or {}).get('label') or (info or {}).get('model')}`"
                for role, info in (model_profile.get("roles") or {}).items()
            )
        )

    task_info = getattr(result, "task", None) or {}
    if task_info.get("task"):
        st.info(
            f"Classified task: **{task_info.get('task')}** "
            f"(confidence {float(task_info.get('confidence') or 0):.2f}, "
            f"via {task_info.get('source')})"
            + (f" — {task_info.get('rationale')}" if task_info.get("rationale") else "")
        )

    model_plan = getattr(result, "model_plan", None) or {}
    probe = getattr(result, "knowledge_probe", None) or {}
    if model_plan.get("enabled"):
        roles = model_plan.get("roles") or {}
        st.success(
            f"Task→model routing **on** · knowledge **{model_plan.get('knowledge_strength')}** "
            f"(hits={probe.get('hit_count', '?')}, coverage={probe.get('lexical_coverage', '?')}) — "
            + ", ".join(
                f"{role}→`{(info or {}).get('label') or (info or {}).get('model')}`"
                for role, info in roles.items()
            )
        )
        with st.expander("Model plan details"):
            st.json(model_plan)
    elif model_plan.get("note") and model_plan.get("note") not in {"not_run", "routing_disabled"}:
        st.caption(f"Task→model routing skipped: {model_plan.get('note')}")

    writebacks = getattr(result, "knowledge_writebacks", None) or []
    if writebacks:
        st.caption(
            f"Knowledge write-back: {len(writebacks)} fact(s) saved for future agents — "
            + ", ".join(f"{w.get('role')}:{w.get('id')}" for w in writebacks[:5])
        )
    compaction = getattr(result, "knowledge_compaction", None) or {}
    if compaction.get("status") == "scheduled":
        st.caption("Knowledge compaction: scheduled in background (does not block this answer).")
    elif compaction.get("deleted"):
        st.caption(
            f"Knowledge compaction: removed {compaction.get('deleted')} redundant doc(s); "
            f"kept `{compaction.get('kept_id') or 'n/a'}`"
        )

    st.subheader("Answer")
    st.write(result.answer)
    st.caption(metrics.answer_quality_notes)

    st.subheader("Per-round / per-agent traces")
    for t in result.traces:
        model_label = getattr(t, "model", "") or ""
        with st.expander(
            f"Round {t.round_idx} · {t.role} · model={model_label or 'n/a'} · "
            f"context_tokens={t.context_tokens}"
        ):
            st.markdown("**Routed context**")
            for i, text in enumerate(t.context_texts, 1):
                st.text(f"[{i}] {text}")
            st.markdown("**Agent output**")
            st.write(t.output)

    with st.expander("Working memory snapshot"):
        st.json(
            [
                {
                    "kind": m.get("struct_kind") or m.get("doc_type"),
                    "key": m.get("struct_key"),
                    "text": m.get("text"),
                }
                for m in result.memory_snapshot
            ]
        )


def page_run() -> None:
    st.header("Run")
    backend = st.session_state.get("knowledge_backend", "unknown")
    wm = working_memory_count() if (_opensearch_enabled() and opensearch_available()) else 0
    thresh = wm_prune_threshold()
    st.caption(
        f"Knowledge store: **{backend}** · write-back "
        f"{'on' if os.getenv('RCR_KNOWLEDGE_WRITEBACK', 'true').lower() in {'1','true','yes'} else 'off'}"
        f" · compact-after-prompt "
        f"**{os.getenv('RCR_COMPACT_AFTER_PROMPT', 'background')}**"
        + (f" · working-memory docs: **{wm}** (auto-clears at ≥{thresh})" if wm else "")
    )
    if wm >= thresh:
        st.warning(
            f"Working memory is large ({wm} docs). "
            "It will be cleared before this run. "
            "Or use **Settings → Clear working memory**."
        )
    store = load_store()
    if store.role_assignments:
        st.success(
            "Using discovered role assignments from Settings: "
            + ", ".join(
                f"{role}→{info.get('label') or info.get('model_id')}"
                + (f" ({info.get('intended_task')})" if info.get("intended_task") else "")
                for role, info in store.role_assignments.items()
                if role in ("planner", "searcher", "recommender")
            )
        )
        low = [
            role
            for role, info in store.role_assignments.items()
            if role in ("planner", "searcher", "recommender")
            and float(info.get("score") or 0) < 0.25
        ]
        if low:
            st.warning(
                "Some discovery scores look low ("
                + ", ".join(low)
                + "). Re-run **Discover capabilities** after fixing API keys — "
                "Model id should be the served name; API key must be a CDP token, not the URL."
            )
    else:
        st.info(
            "No capability discovery yet. Open **Settings**, add up to 3 LLM endpoints, "
            "then run **Discover capabilities** so the app can pick the best model per agent."
        )

    query = st.text_area(
        "User query",
        value="What drove Acme Corp revenue growth in Q3?",
        height=80,
    )
    cols = st.columns(5)
    mode = cols[0].selectbox("Mode", ["single", "compare all"], index=1)
    strategy = cols[1].selectbox("Strategy", ["rcr", "static", "full"], index=0)
    max_rounds = cols[2].slider("Rounds (T)", 1, 5, 2)
    preset_options = ["savings", "paper", "demo"]
    env_preset = os.getenv("RCR_BUDGET_PRESET", "savings").strip().lower()
    preset_index = (
        preset_options.index(env_preset) if env_preset in preset_options else 0
    )
    budget_preset_ui = cols[3].selectbox(
        "Budget preset",
        preset_options,
        index=preset_index,
        help=(
            "savings (default): tighter B_i for clear RCR token cuts vs full. "
            "paper: §2-scale bands. demo: tiny local/unit budgets."
        ),
    )
    run = cols[4].button("Run", type="primary")
    use_judge = st.checkbox(
        "LLM answer judge (paper Appendix A.4)",
        value=os.getenv("RCR_LLM_JUDGE", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Optional 1–5 judge score via the active LLM. Off by default so smoke/compare stays fast.",
    )
    use_task_model_routing = st.checkbox(
        "Route task to models (knowledge-aware)",
        value=os.getenv("RCR_TASK_MODEL_ROUTING", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Opt-in: classify the query, probe the knowledge store, then bind "
            "Planner/Searcher/Recommender to matching Settings endpoints. "
            "Off by default. Needs ≥2 configured endpoints."
        ),
    )

    if not (run and query.strip()):
        return

    os.environ["RCR_BUDGET_PRESET"] = budget_preset_ui
    orch = get_orchestrator(force_reload=True)
    q = query.strip()
    pruned = prune_working_memory_if_needed()
    if pruned:
        st.caption(f"Pruned {pruned} stale working-memory doc(s) before run (keeps latency down).")

    judge_llm = orch.llm if use_judge else None

    if mode == "compare all":
        with st.spinner("Comparing full / static / rcr…"):
            rows = []
            results = {}
            # Disable write-back during compare so later strategies aren't polluted
            # by earlier ones writing into the shared knowledge index.
            for strat in ("full", "static", "rcr"):
                r = orch.run(
                    q,
                    strategy=RoutingStrategy(strat),
                    max_rounds=max_rounds,
                    knowledge_writeback=False,
                    task_model_routing=use_task_model_routing,
                )
                results[strat] = r
                rows.append(
                    summarize_run(r, q, llm=judge_llm, use_judge=use_judge).to_dict()
                )
        st.subheader("Strategy comparison")
        st.caption(
            "Fair compare: knowledge write-back off. Prefer **prompt tokens** and "
            "**context tokens** for RCR savings — Nemotron completion/reasoning can "
            "dominate total LLM tokens. "
            + (
                "**Lexical** + **judge** quality both shown when LLM judge is on."
                if use_judge
                else "Lexical quality only (enable LLM judge for paper-style scores)."
            )
            + (
                " Task→model routing **on** for this compare."
                if use_task_model_routing
                else ""
            )
        )
        st.dataframe(rows, width="stretch")
        tabs = st.tabs(["rcr", "static", "full"])
        for tab, strat in zip(tabs, ("rcr", "static", "full")):
            with tab:
                render_result(results[strat], q, use_judge=use_judge, llm=judge_llm)
    else:
        with st.spinner("Running multi-agent routing…"):
            result = orch.run(
                q,
                strategy=RoutingStrategy(strategy),
                max_rounds=max_rounds,
                task_model_routing=use_task_model_routing,
            )
        render_result(result, q, use_judge=use_judge, llm=judge_llm)


def page_settings() -> None:
    st.header("Settings · LLM endpoints")
    st.caption(
        "Add up to **3** OpenAI-compatible endpoints (Cloudera AI Inference / NIM, etc.). "
        "Set each model's **intended task** (generic, coding, …), then run capability discovery "
        "so we can assign Planner / Searcher / Recommender / Router roles."
    )
    st.caption(f"Stored at `{default_store_path()}` (local; not committed).")

    store = load_store()
    if len(store.endpoints) < 3:
        store.endpoints = blank_endpoints()

    task_options = list(INTENDED_TASKS)
    edited: list[UserEndpoint] = []
    for i, ep in enumerate(store.endpoints[:3]):
        task_default = ep.normalized_task()
        task_idx = task_options.index(task_default) if task_default in task_options else 0
        with st.expander(
            f"Endpoint {ep.slot}: {ep.label or ep.model_id or 'empty'} · {task_default}",
            expanded=True,
        ):
            c1, c2 = st.columns(2)
            label = c1.text_input("Label", value=ep.label, key=f"label_{i}")
            enabled = c2.checkbox("Enabled", value=ep.enabled, key=f"en_{i}")
            intended_task = st.selectbox(
                "Intended task",
                options=task_options,
                index=task_idx,
                format_func=lambda t: INTENDED_TASK_LABELS.get(t, t),
                key=f"task_{i}",
                help="What you bought this model for — biases capability priors before live probes.",
            )
            model_id = st.text_input(
                "Model id",
                value=ep.model_id,
                placeholder="nvidia/nemotron-3-super-120b-a12b",
                key=f"mid_{i}",
                help="Registry / served model name used in chat completions.",
            )
            api_base = st.text_input(
                "API base (…/v1)",
                value=ep.api_base,
                placeholder="https://…/endpoints/<name>/v1",
                key=f"base_{i}",
            )
            api_key = st.text_input(
                "API key / CDP token",
                value=ep.api_key,
                type="password",
                key=f"key_{i}",
                help="CDP JWT / bearer token only — not the endpoint URL.",
            )
            if api_key.strip().startswith("http://") or api_key.strip().startswith("https://"):
                st.error(
                    f"Endpoint {ep.slot}: API key looks like a URL. Paste the CDP token instead."
                )
            edited.append(
                UserEndpoint(
                    slot=ep.slot,
                    label=label.strip() or f"Model {ep.slot}",
                    model_id=model_id.strip(),
                    api_base=api_base.strip().rstrip("/"),
                    api_key=api_key.strip(),
                    intended_task=intended_task,
                    enabled=enabled,
                )
            )

    b1, b2, b3 = st.columns(3)
    if b1.button("Save endpoints", type="primary"):
        store.endpoints = edited
        save_store(store)
        st.success("Saved endpoints.")
        st.session_state.pop("orch", None)

    use_mock_probe = b2.checkbox(
        "Dry-run discovery (mock LLM)",
        value=False,
        help="Score without calling real endpoints — for UI/testing only.",
    )

    if b3.button("Discover capabilities"):
        store.endpoints = edited
        save_store(store)
        configured = [e for e in edited if e.enabled and e.is_configured()]
        if not configured:
            st.error("Configure at least one enabled endpoint with model id + api base.")
        else:
            with st.spinner(
                f"Probing {len(configured)} model(s) for planner / searcher / "
                "recommender / router skills…"
            ):
                store = run_discovery(store, use_mock=use_mock_probe)
            st.session_state.pop("orch", None)
            st.success(
                f"Discovery complete at {store.last_probed_at}. "
                "Role assignments updated."
            )

    store = load_store()
    if store.capabilities:
        st.subheader("Capability scores")
        rows = []
        for ep in store.configured_endpoints():
            cap = store.capabilities.get(ep.model_id) or {}
            rows.append(
                {
                    "slot": ep.slot,
                    "label": ep.label,
                    "intended_task": ep.normalized_task(),
                    "model_id": ep.model_id,
                    "planner": cap.get("planner"),
                    "searcher": cap.get("searcher"),
                    "recommender": cap.get("recommender"),
                    "router": cap.get("router"),
                    "latency_sec": cap.get("latency_sec"),
                }
            )
        st.dataframe(rows, width="stretch")

        with st.expander("Probe notes / samples"):
            for ep in store.configured_endpoints():
                cap = store.capabilities.get(ep.model_id) or {}
                st.markdown(f"**{ep.label or ep.model_id}**")
                st.json(
                    {
                        "notes": cap.get("notes"),
                        "samples": cap.get("samples"),
                    }
                )

    if store.role_assignments:
        st.subheader("Assigned agents (best-fit)")
        st.dataframe(
            [
                {
                    "role": role,
                    "slot": info.get("slot"),
                    "label": info.get("label"),
                    "intended_task": info.get("intended_task"),
                    "model_id": info.get("model_id"),
                    "score": info.get("score"),
                }
                for role, info in store.role_assignments.items()
            ],
            width="stretch",
        )

    st.subheader("Embeddings (P4)")
    st.caption(
        "Chat (Nemotron) and embed (`llama-3.2-nv-embedqa`) are separate CAIIS endpoints. "
        f"Env: `RCR_USE_MOCK_LLM={os.getenv('RCR_USE_MOCK_LLM')}` · "
        f"`RCR_EMBED_BACKEND={os.getenv('RCR_EMBED_BACKEND')}` · "
        f"embed_model=`{os.getenv('CLOUDERA_AI_INFERENCE_EMBED_MODEL') or 'unset'}`"
    )
    emb_cols = st.columns(3)
    if emb_cols[0].button("Probe embeddings", key="probe_embed_btn"):
        with st.spinner("Probing embeddings…"):
            orch = get_orchestrator(force_reload=True)
            client_type = type(orch.llm).__name__
            status = probe_embeddings(orch.llm)
        payload = status.to_dict()
        payload["client_type"] = client_type
        st.session_state["last_embed_status"] = payload
        if status.ok:
            st.success(
                f"client=**{client_type}** backend=**{status.backend}** dim=**{status.dim}** "
                f"model=`{status.model or 'n/a'}`"
                + (f" — {status.detail}" if status.detail else "")
            )
            if client_type == "MockLLMClient" or status.backend == "mock":
                st.warning(
                    "Orchestrator is on **MockLLMClient** (or mock embed backend). "
                    "Set `RCR_USE_MOCK_LLM=false` in `.env`, restart the Streamlit app, "
                    "then Probe again. EmbedQA is configured via "
                    "`CLOUDERA_AI_INFERENCE_EMBED_*`."
                )
            elif str(status.backend).startswith("local"):
                st.info(
                    "Local semantic embeddings are active. "
                    "After a dim change, run **Sync indices** then **Re-seed knowledge**."
                )
            elif status.backend == "litellm":
                st.info("Live CAIIS embeddings OK. If dim changed, **Sync indices** + **Re-seed**.")
        else:
            st.error(f"Embed probe failed: {status.detail}")

    if emb_cols[1].button("Sync indices to embed dim", key="sync_embed_dim_btn"):
        with st.spinner("Probing + syncing OpenSearch knn dim…"):
            orch = get_orchestrator(force_reload=True)
            status = prepare_embeddings(orch.llm, recreate_on_mismatch=True)
        st.session_state["last_embed_status"] = status.to_dict()
        if status.ok:
            st.success(
                f"Synced dim={status.dim} (index_dim={status.index_dim}). {status.detail}"
            )
            st.info("Re-seed knowledge so corpus vectors match the new dimension.")
        else:
            st.error(status.detail or "sync failed")

    if emb_cols[2].button("Re-seed knowledge", key="reseed_knowledge_btn"):
        with st.spinner("Seeding knowledge corpus with current embed backend…"):
            orch = get_orchestrator(force_reload=True)
            prepare_embeddings(orch.llm, recreate_on_mismatch=True)
            seed_knowledge(orch)
            st.session_state["knowledge_backend"] = (
                "opensearch" if _opensearch_enabled() else "in-memory"
            )
        st.success(
            f"Seeded knowledge. embed_backend="
            f"{getattr(orch.llm, 'last_embed_backend', '?')} "
            f"dim={os.getenv('OPENSEARCH_EMBED_DIM')}"
        )

    last_emb = st.session_state.get("last_embed_status")
    if last_emb:
        st.caption(
            f"Last probe: backend={last_emb.get('backend')} dim={last_emb.get('dim')} "
            f"mismatch={last_emb.get('dim_mismatch')} — {last_emb.get('detail') or ''}"
        )

    st.subheader("Index hygiene")
    backend = st.session_state.get("knowledge_backend")
    if not backend:
        backend = "opensearch" if _opensearch_enabled() else "in-memory"
    auto_compact = os.getenv("RCR_COMPACT_AFTER_PROMPT", "background").strip().lower()
    wm_n = working_memory_count() if (_opensearch_enabled() and opensearch_available()) else 0
    thresh = wm_prune_threshold()

    st.caption(
        f"Store: **{backend}**. Two different indices: "
        "**working memory** (per-run agent traces — clear when large) vs "
        "**knowledge** (seeded corpus + query writebacks — compact duplicates)."
    )
    st.caption(
        f"Compact-after-prompt mode: **{auto_compact or 'background'}** "
        f"(`RCR_COMPACT_AFTER_PROMPT` = `background` | `sync` | `off`). "
        "**Compact knowledge** also removes low-value writebacks (refusals, plan/meta chrome)."
    )

    # Auto-clear bloated WM when opening Settings (same threshold as Run).
    if wm_n >= thresh:
        pruned = prune_working_memory_if_needed()
        if pruned:
            st.success(
                f"Auto-cleared **{pruned}** working-memory doc(s) "
                f"(was {wm_n}, threshold {thresh}). Knowledge corpus kept."
            )
            wm_n = working_memory_count() if (_opensearch_enabled() and opensearch_available()) else 0
        else:
            st.warning(
                f"Working memory has **{wm_n}** docs (threshold {thresh}). "
                "This slows search / elapsed time — use **Clear working memory** below. "
                "Knowledge compaction will not shrink this index."
            )
    elif wm_n:
        st.caption(f"Working-memory docs: **{wm_n}** (clears automatically at ≥{thresh}).")

    c1, c2 = st.columns(2)
    if c1.button("Compact knowledge now", key="compact_knowledge_btn"):
        with st.spinner("Compacting knowledge corpus…"):
            orch = get_orchestrator()
            wb = KnowledgeWriteback(orch.knowledge)
            result = wb.compact_corpus()
        st.session_state["last_compaction"] = result.as_dict()
        if result.deleted:
            st.success(
                f"Knowledge: removed {result.deleted} redundant writeback/task doc(s) across "
                f"{result.queries_compacted} query slot(s). "
                f"{result.remaining_writebacks} canonical writeback(s) kept."
            )
        else:
            st.info(
                f"Knowledge already clean: scanned {result.queries_compacted} query slot(s); "
                f"{result.remaining_writebacks} writeback(s) kept "
                "(one per distinct query is expected). "
                "This does **not** clear working memory."
            )
            if wm_n >= thresh:
                st.warning(
                    f"Working memory still has {wm_n} docs — click **Clear working memory**."
                )

    if c2.button("Clear working memory", key="clear_wm_btn", type="primary"):
        with st.spinner("Clearing working-memory index…"):
            deleted = prune_working_memory_if_needed(force=True)
        st.success(
            f"Cleared **{deleted}** working-memory doc(s). "
            "Knowledge corpus (seeds + writebacks) kept."
        )

    last = st.session_state.get("last_compaction")
    if last:
        with st.expander("Last knowledge compaction details", expanded=bool(last.get("details"))):
            st.write(
                {
                    "deleted": last.get("deleted"),
                    "queries_compacted": last.get("queries_compacted"),
                    "remaining_writebacks": last.get("remaining_writebacks"),
                }
            )
            if last.get("details"):
                st.dataframe(last["details"], width="stretch")
            else:
                st.caption("No per-slot deletions — each query already had a single writeback.")


def main() -> None:
    st.set_page_config(page_title="RCR — Role-Based Agent Routing", layout="wide")
    st.title("RCR — Role-Based Agent Routing")
    st.caption(
        "**RCR** (*Role-aware Context Routing*) gives each agent a scored, token-budgeted "
        "memory slice—not the full history. OpenSearch · LiteLLM · optional task→model routing."
    )

    page = st.sidebar.radio("Navigation", ["Run", "Settings"], index=0)
    if page == "Settings":
        page_settings()
    else:
        page_run()


if __name__ == "__main__":
    main()
