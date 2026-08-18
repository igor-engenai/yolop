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
        "deferred_tool_requests",
        "deferred_tool_results",
        "enqueued_messages",
        "final_result",
        "function_tool_call",
        "function_tool_result",
        "output_tool_call",
        "output_tool_result",
        "part_end",
        "part_start",
        "realtime_input_speech_end",
        "realtime_input_speech_start",
        "realtime_input_transcription_error",
        "realtime_output_speech_end",
        "realtime_output_speech_start",
        "realtime_response_interrupted",
        "realtime_session_error",
        "realtime_session_reconnect",
        "realtime_turn_complete",
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
