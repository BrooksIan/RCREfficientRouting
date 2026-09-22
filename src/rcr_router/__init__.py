"""RCR-Router: role-aware, token-budgeted context routing for multi-agent LLMs."""

from .budget import TokenBudgetAllocator
from .models import AgentRole, MemoryItem, RoutingStrategy, TaskStage
from .orchestrator import Orchestrator
from .router import RCRRouter

__all__ = [
    "AgentRole",
    "MemoryItem",
    "Orchestrator",
    "RCRRouter",
    "RoutingStrategy",
    "TaskStage",
    "TokenBudgetAllocator",
]

__version__ = "0.2.0"
