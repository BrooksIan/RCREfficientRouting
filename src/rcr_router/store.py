from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .models import MemoryItem


class MemoryStore(ABC):
    """Working memory M_t (session-scoped)."""

    @abstractmethod
    def upsert(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def bulk_upsert(self, items: list[MemoryItem]) -> None: ...

    @abstractmethod
    def list_session(self, session_id: str) -> list[MemoryItem]: ...

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        role_tag: Optional[str] = None,
        stage_tag: Optional[str] = None,
        task: Optional[str] = None,
        k: int = 20,
        embedding: Optional[list[float]] = None,
    ) -> list[MemoryItem]: ...

    def delete(self, item_id: str) -> None:
        """Optional: remove a working-memory doc by id."""
        return None


class KnowledgeStore(ABC):
    """Semantic knowledge corpus for the Searcher."""

    @abstractmethod
    def upsert(self, item: MemoryItem) -> None: ...

    @abstractmethod
    def bulk_upsert(self, items: list[MemoryItem]) -> None: ...

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        embedding: Optional[list[float]] = None,
        task: Optional[str] = None,
        soft_task: bool = True,
    ) -> list[MemoryItem]: ...

    def delete(self, item_id: str) -> None:
        """Optional: remove a document by id (OpenSearch / in-memory)."""
        return None
