from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic_ai import AgentStreamEvent


class RuntimeEventPersistencePolicy(Protocol):
    """Select native events that the durable RuntimeStore should retain."""

    def should_persist(self, event: AgentStreamEvent) -> bool: ...


@dataclass(frozen=True)
class AllEventsPersistencePolicy:
    """Persist every native event while still delivering it to live sinks."""

    def should_persist(self, event: AgentStreamEvent) -> bool:
        del event
        return True


_DEFAULT_SEMANTIC_EVENT_KINDS = frozenset(
    {
        "agent_run_started",
        "agent_run_result",
        "builtin_tool_call",
        "builtin_tool_result",
        "deferred_tool_call",
        "deferred_tool_result",
        "function_tool_call",
        "function_tool_result",
        "run_cancelled",
        "run_completed",
        "run_failed",
        "run_interrupted",
        "tool_call",
        "tool_result",
    }
)


@dataclass(frozen=True)
class SparseEventPersistencePolicy:
    """Persist selected semantic events and drop high-volume stream deltas."""

    semantic_event_kinds: frozenset[str] = field(default=_DEFAULT_SEMANTIC_EVENT_KINDS)

    def should_persist(self, event: AgentStreamEvent) -> bool:
        return event.event_kind in self.semantic_event_kinds


__all__ = [
    "AllEventsPersistencePolicy",
    "RuntimeEventPersistencePolicy",
    "SparseEventPersistencePolicy",
]
