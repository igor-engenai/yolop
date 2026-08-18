from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from yolop_runtime import RuntimeEventSink, RuntimeStore, SparseEventPersistencePolicy
from yolop_runtime.runtime import _event_handler


@dataclass(frozen=True)
class Event:
    event_kind: str


class Store:
    def __init__(self) -> None:
        self.persisted: list[str] = []

    async def append_run_event(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        event: str,
        data: str,
    ) -> None:
        del namespace, run_id, owner_id, data
        self.persisted.append(event)


class Adapter:
    def dump_json(self, event: Event) -> bytes:
        return event.event_kind.encode()


class Sink:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event.event_kind)


def test_sparse_policy_uses_current_semantic_event_kinds() -> None:
    policy = SparseEventPersistencePolicy()

    for event_kind in (
        "deferred_tool_requests",
        "deferred_tool_results",
        "final_result",
        "function_tool_call",
        "function_tool_result",
        "output_tool_call",
        "output_tool_result",
        "part_start",
        "part_end",
    ):
        assert policy.should_persist(cast(Any, Event(event_kind)))
    assert not policy.should_persist(cast(Any, Event("part_delta")))


@pytest.mark.asyncio
async def test_sparse_policy_keeps_live_events_and_persists_semantic_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("yolop_runtime.runtime._STREAM_EVENT_ADAPTER", Adapter())
    store_impl = Store()
    sink_impl = Sink()
    store = cast(RuntimeStore, store_impl)
    sink = cast(RuntimeEventSink, sink_impl)

    async def events():
        for event_kind in ("text_delta", "tool_call", "run_completed"):
            yield Event(event_kind)

    handler = _event_handler(
        store,
        namespace="tenant/acme",
        run_id="run",
        owner_id="worker",
        event_sink=sink,
        context_sink=None,
        event_persistence_policy=SparseEventPersistencePolicy(
            semantic_event_kinds=frozenset({"tool_call", "run_completed"})
        ),
    )

    await handler(cast(Any, None), cast(Any, events()))

    assert sink_impl.events == ["text_delta", "tool_call", "run_completed"]
    assert store_impl.persisted == ["tool_call", "run_completed"]
