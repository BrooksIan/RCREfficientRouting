"""Memory Update pipeline: extract → filter → structure → conflict resolve → upsert."""

from __future__ import annotations

import time
from typing import Iterable

from .models import AgentRole, DocType, MemoryItem, TaskStage
from .store import MemoryStore
from .structured_memory import (
    StructuredUnit,
    extract_structured_units,
    to_memory_text,
)


class MemoryUpdater:
    """Paper Memory Update: extract → filter → structure → conflict resolve."""

    def __init__(self, memory: MemoryStore, min_novelty_chars: int = 12) -> None:
        self.memory = memory
        self.min_novelty_chars = min_novelty_chars

    def get_by_key(self, session_id: str, key: str) -> MemoryItem | None:
        key = (key or "").strip()
        if not key:
            return None
        for item in self.memory.list_session(session_id):
            if (item.struct_key or "") == key:
                return item
            meta_key = (item.metadata or {}).get("struct_key")
            if meta_key == key:
                return item
        return None

    def list_by_kind(
        self,
        session_id: str,
        kind: str,
        *,
        role_tag: str | None = None,
    ) -> list[MemoryItem]:
        kind = (kind or "").strip()
        out: list[MemoryItem] = []
        for item in self.memory.list_session(session_id):
            if role_tag and item.role_tag != role_tag:
                continue
            item_kind = item.struct_kind or (item.metadata or {}).get("struct_kind")
            if item_kind == kind:
                out.append(item)
        out.sort(key=lambda m: (m.round, m.timestamp))
        return out

    def update_from_agent_output(
        self,
        *,
        text: str,
        role: AgentRole,
        stage: TaskStage,
        session_id: str,
        round_idx: int,
        doc_type: str = DocType.INTERACTION.value,
        embedding: list[float] | None = None,
        existing: Iterable[MemoryItem] | None = None,
        task: str | None = None,
    ) -> list[MemoryItem]:
        cleaned = (text or "").strip()
        if len(cleaned) < self.min_novelty_chars:
            return []

        existing_list = list(existing) if existing is not None else self.memory.list_session(
            session_id
        )
        existing_texts = {m.text.strip().lower() for m in existing_list if m.text}

        units = extract_structured_units(cleaned, role=role)
        if not units:
            return []

        # Exact-text novelty: skip units whose YAML/text already exists
        written: list[MemoryItem] = []
        plan_keys_written: set[str] = set()

        for unit in units:
            item = self._unit_to_item(
                unit,
                role=role,
                stage=stage,
                session_id=session_id,
                round_idx=round_idx,
                embedding=embedding,
                task=task,
                fallback_doc_type=doc_type,
            )
            if item.text.strip().lower() in existing_texts:
                # Same prose already stored — still allow key overwrite if content moved keys
                prior = self._find_key(existing_list, unit.key)
                if prior and prior.text.strip().lower() == item.text.strip().lower():
                    continue

            item = self._resolve_conflict(item, existing_list, unit)
            self.memory.upsert(item)
            written.append(item)
            existing_list.append(item)
            existing_texts.add(item.text.strip().lower())
            if unit.kind == "plan_step":
                plan_keys_written.add(unit.key)

        # Supersede leftover plan steps when a shorter revised plan arrives
        if role == AgentRole.PLANNER and plan_keys_written:
            self._prune_superseded_plans(
                session_id, keep_keys=plan_keys_written, existing=existing_list
            )

        return written

    def _unit_to_item(
        self,
        unit: StructuredUnit,
        *,
        role: AgentRole,
        stage: TaskStage,
        session_id: str,
        round_idx: int,
        embedding: list[float] | None,
        task: str | None,
        fallback_doc_type: str,
    ) -> MemoryItem:
        yaml_text = to_memory_text(unit)
        structure = {
            "kind": unit.kind,
            "key": unit.key,
            **unit.payload,
        }
        meta: dict = {
            "struct_key": unit.key,
            "struct_kind": unit.kind,
            "structured": True,
        }
        if task:
            meta["task"] = task
        if unit.payload.get("evidence_ids"):
            meta["evidence_ids"] = unit.payload["evidence_ids"]
        return MemoryItem(
            text=yaml_text,
            role_tag=role.value,
            stage_tag=stage.value,
            source_agent=role.value,
            session_id=session_id,
            round=round_idx,
            timestamp=time.time(),
            doc_type=unit.doc_type or fallback_doc_type,
            embedding=embedding,
            task=task,
            struct_key=unit.key,
            struct_kind=unit.kind,
            structure=structure,
            metadata=meta,
        )

    def _find_key(self, items: Iterable[MemoryItem], key: str) -> MemoryItem | None:
        for item in items:
            if (item.struct_key or "") == key:
                return item
            if (item.metadata or {}).get("struct_key") == key:
                return item
        return None

    def _resolve_conflict(
        self,
        item: MemoryItem,
        existing: list[MemoryItem],
        unit: StructuredUnit,
    ) -> MemoryItem:
        """Same struct_key → replace in place (reuse id); mark prior superseded."""
        prior = self._find_key(existing, unit.key)
        if prior is None:
            return item
        # Prefer newer content; reuse id so stores overwrite the same doc
        item.id = prior.id
        meta = dict(item.metadata or {})
        meta["supersedes"] = prior.id
        meta["prior_round"] = prior.round
        item.metadata = meta
        # Drop stale entry from working list (store already overwritten on upsert)
        try:
            existing.remove(prior)
        except ValueError:
            pass
        return item

    def _prune_superseded_plans(
        self,
        session_id: str,
        *,
        keep_keys: set[str],
        existing: list[MemoryItem],
    ) -> None:
        """Delete plan_step items whose step index is no longer in the latest plan."""
        for item in list(existing):
            if item.session_id != session_id:
                continue
            kind = item.struct_kind or (item.metadata or {}).get("struct_kind")
            key = item.struct_key or (item.metadata or {}).get("struct_key") or ""
            if kind != "plan_step" or not key:
                continue
            if key in keep_keys:
                continue
            delete = getattr(self.memory, "delete", None)
            if callable(delete):
                delete(item.id)
            try:
                existing.remove(item)
            except ValueError:
                pass
