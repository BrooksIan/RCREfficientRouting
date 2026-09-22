from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional
import time
import uuid


class AgentRole(str, Enum):
    PLANNER = "planner"
    SEARCHER = "searcher"
    RECOMMENDER = "recommender"


class TaskStage(str, Enum):
    PLANNING = "planning"
    SEARCHING = "searching"
    RECOMMENDING = "recommending"


class RoutingStrategy(str, Enum):
    FULL = "full"
    STATIC = "static"
    RCR = "rcr"


class DocType(str, Enum):
    INTERACTION = "interaction"
    FACT = "fact"
    PLAN = "plan"
    TOOL_TRACE = "tool_trace"
    KNOWLEDGE = "knowledge"


@dataclass
class MemoryItem:
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role_tag: Optional[str] = None
    stage_tag: Optional[str] = None
    source_agent: Optional[str] = None
    session_id: Optional[str] = None
    round: int = 0
    timestamp: float = field(default_factory=time.time)
    token_length: int = 0
    doc_type: str = DocType.INTERACTION.value
    embedding: Optional[list[float]] = None
    score: float = 0.0
    task: Optional[str] = None  # prompt/knowledge task label (shared taxonomy)
    struct_key: Optional[str] = None  # stable key for conflict resolve (P2)
    struct_kind: Optional[str] = None  # fact | plan_step | evidence | answer | …
    structure: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_length:
            from .tokens import count_tokens

            self.token_length = count_tokens(self.text or "")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        return cls(**payload)


ROLE_KEYWORDS: dict[AgentRole, list[str]] = {
    AgentRole.PLANNER: ["plan", "decompose", "intent", "strategy", "subquestion", "query"],
    AgentRole.SEARCHER: ["evidence", "fact", "retrieve", "document", "search", "passage"],
    AgentRole.RECOMMENDER: ["answer", "summary", "recommend", "conclude", "final"],
}

STAGE_FOR_ROLE: dict[AgentRole, TaskStage] = {
    AgentRole.PLANNER: TaskStage.PLANNING,
    AgentRole.SEARCHER: TaskStage.SEARCHING,
    AgentRole.RECOMMENDER: TaskStage.RECOMMENDING,
}
