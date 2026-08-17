from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from pydantic_ai import AgentSpec
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage
from pytest import raises
from yolop_runtime import (
    ExecutionPin,
    PluginStateEntry,
    RunCompletion,
    RunRelation,
    RunReservation,
    RunStatus,
    Runtime,
    RuntimeRunSnapshot,
    RuntimeSessionSnapshot,
    ScopedStateContext,
    SessionPinMismatchError,
    StateScope,
    StateSequenceConflictError,
)

from yolop import ProviderCatalog


@dataclass
class MemoryStore:
    session: RuntimeSessionSnapshot | None = None
    run: RuntimeRunSnapshot | None = None
    state_entries: list[PluginStateEntry] = field(default_factory=list)

    async def create_session(self, namespace: str, *, pin: ExecutionPin) -> RuntimeSessionSnapshot:
        self.session = RuntimeSessionSnapshot(
            id="00000000-0000-4000-8000-000000000001",
            namespace=namespace,
            pin=pin,
            messages=[],
            revision="empty",
        )
        return self.session

    async def load_session(self, namespace: str, session_id: str) -> RuntimeSessionSnapshot:
        assert self.session is not None
        assert (namespace, session_id) == (self.session.namespace, self.session.id)
        return self.session

    async def list_sessions(self, namespace: str) -> list[str]:
        del namespace
        return [] if self.session is None else [self.session.id]

    async def delete_session(
        self,
        namespace: str,
        session_id: str,
        *,
        expected_revision: str,
    ) -> None:
        del namespace, session_id, expected_revision

    async def load_root_budget(
        self,
        namespace: str,
        root_run_id: str,
    ):
        del namespace, root_run_id
        return None

    async def read_state(
        self,
        namespace: str,
        *,
        owner_id: str,
        scope: StateScope,
        scope_id: str,
        state_kind: str,
        schema_version: int | None = None,
    ) -> list[PluginStateEntry]:
        del namespace
        entries = [
            entry
            for entry in self.state_entries
            if entry.owner_id == owner_id
            and entry.scope is scope
            and entry.scope_id == scope_id
            and entry.state_kind == state_kind
        ]
        if schema_version is not None:
            entries = [entry for entry in entries if entry.schema_version == schema_version]
        return entries

    async def append_state(
        self,
        namespace: str,
        *,
        owner_id: str,
        scope: StateScope,
        scope_id: str,
        state_kind: str,
        schema_version: int,
        expected_sequence: int,
        payload: Any,
    ) -> PluginStateEntry:
        del namespace
        current_sequence = max(
            (
                entry.sequence
                for entry in self.state_entries
                if entry.owner_id == owner_id
                and entry.scope is scope
                and entry.scope_id == scope_id
                and entry.state_kind == state_kind
            ),
            default=0,
        )
        if current_sequence != expected_sequence:
            raise StateSequenceConflictError("stale state sequence")
        entry = PluginStateEntry(
            namespace="test",
            owner_id=owner_id,
            scope=scope,
            scope_id=scope_id,
            state_kind=state_kind,
            schema_version=schema_version,
            sequence=expected_sequence + 1,
            payload=payload,
            created_at=datetime.now(UTC),
        )
        self.state_entries.append(entry)
        return entry

    async def replace_session(
        self,
        namespace: str,
        session_id: str,
        *,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> RuntimeSessionSnapshot:
        del namespace, session_id, expected_revision
        assert self.session is not None
        self.session = replace(self.session, messages=list(messages), revision="replaced")
        return self.session

    async def reserve_run(
        self,
        namespace: str,
        session_id: str,
        *,
        idempotency_key: str,
        prompt: str,
        max_pending: int | None = None,
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
        relation: RunRelation = RunRelation.ROOT,
        root_budget: Any = None,
        initiator: str = "user",
        input_digest: str | None = None,
        full_messages: Sequence[ModelMessage] = (),
        active_messages: Sequence[ModelMessage] = (),
    ) -> RunReservation:
        del max_pending, root_budget, input_digest
        now = datetime.now(UTC)
        self.run = RuntimeRunSnapshot(
            id="00000000-0000-4000-8000-000000000002",
            namespace=namespace,
            session_id=session_id,
            idempotency_key=idempotency_key,
            prompt=prompt,
            status=RunStatus.ACCEPTED,
            created_at=now,
            updated_at=now,
            parent_run_id=parent_run_id,
            root_run_id=root_run_id or parent_run_id or "00000000-0000-4000-8000-000000000002",
            relation=relation,
            initiator=initiator,
            full_messages=list(full_messages),
            active_messages=list(active_messages),
        )
        return RunReservation(self.run, created=True)

    async def load_run(self, namespace: str, run_id: str) -> RuntimeRunSnapshot:
        del namespace, run_id
        assert self.run is not None
        return self.run

    async def list_runs(
        self,
        namespace: str,
        *,
        session_id: str | None = None,
    ) -> list[RuntimeRunSnapshot]:
        del namespace, session_id
        return [] if self.run is None else [self.run]

    async def cancel_run(
        self,
        namespace: str,
        run_id: str,
        *,
        error_code: str = "run_cancelled",
        error_detail: str = "Run cancelled",
        expected_session_revision: str | None = None,
        full_messages: Sequence[ModelMessage] | None = None,
        active_messages: Sequence[ModelMessage] | None = None,
        output: Any | None = None,
        usage: RunUsage | None = None,
    ) -> RuntimeRunSnapshot:
        del namespace, run_id, expected_session_revision
        assert self.run is not None
        self.run = replace(
            self.run,
            status=RunStatus.INTERRUPTED,
            error_code=error_code,
            error_detail=error_detail,
            full_messages=list(full_messages or self.run.full_messages),
            active_messages=list(active_messages or self.run.active_messages),
            output=output,
            usage=usage,
        )
        assert self.session is not None
        self.session = replace(
            self.session,
            messages=list(self.run.active_messages),
            revision="cancelled",
            head_run_id=self.run.id,
        )
        return self.run

    async def claim_run(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot:
        del namespace, run_id, lease_seconds
        assert self.run is not None
        self.run = replace(self.run, status=RunStatus.RUNNING, owner_id=owner_id)
        return self.run

    @asynccontextmanager
    async def lock_session(
        self, namespace: str, session_id: str, *, timeout: float
    ) -> AsyncIterator[None]:
        del namespace, session_id, timeout
        yield

    async def list_run_events(self, namespace: str, run_id: str, *, after: int = 0):
        del namespace, run_id, after
        return []

    async def interrupt_owned_runs(self, owner_id: str) -> int:
        del owner_id
        return 0

    async def interrupt_expired_runs(self) -> int:
        return 0

    async def renew_run_lease(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot:
        del namespace, run_id, owner_id, lease_seconds
        assert self.run is not None
        return self.run

    async def checkout_session(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        *,
        expected_revision: str,
    ) -> RuntimeSessionSnapshot:
        del namespace, session_id, expected_revision
        assert self.session is not None and self.run is not None
        assert self.run.id == run_id
        self.session = replace(
            self.session,
            messages=list(self.run.active_messages),
            revision="checkout",
            head_run_id=run_id,
        )
        return self.session

    async def append_run_event(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        event: str,
        data: str,
    ):
        del namespace, run_id, owner_id, event, data
        return None

    async def complete_run(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        expected_session_revision: str,
        messages: Sequence[ModelMessage] | None = None,
        full_messages: Sequence[ModelMessage] | None = None,
        active_messages: Sequence[ModelMessage] | None = None,
        output: Any,
        usage: RunUsage,
    ) -> RunCompletion:
        del namespace, run_id, owner_id, expected_session_revision, usage
        assert self.session is not None and self.run is not None
        full_messages = list(full_messages if full_messages is not None else messages or ())
        active_messages = list(active_messages if active_messages is not None else messages or ())
        self.session = replace(
            self.session,
            messages=active_messages,
            revision="complete",
            head_run_id=self.run.id,
        )
        self.run = replace(
            self.run,
            status=RunStatus.COMPLETED,
            output=output,
            full_messages=full_messages,
            active_messages=active_messages,
        )
        return RunCompletion(self.session, self.run)

    async def fail_run(self, *args, **kwargs):
        raise AssertionError((args, kwargs))


@dataclass
class EventSink:
    events: list[object]

    async def emit(self, event: object) -> None:
        self.events.append(event)


def test_execution_scope_defaults_root_run_and_rejects_empty_initiator() -> None:
    from yolop_runtime import ExecutionScope

    scope = ExecutionScope(
        namespace="test",
        session_id="00000000-0000-4000-8000-000000000001",
        run_id="00000000-0000-4000-8000-000000000002",
    )

    assert scope.root_run_id == scope.run_id
    with raises(ValueError, match="initiator"):
        ExecutionScope(
            namespace="test",
            session_id=scope.session_id,
            run_id=scope.run_id,
            initiator=" ",
        )


async def test_memory_store_state_contract_is_owner_bound_and_compare_append() -> None:
    store = MemoryStore()
    session = await store.create_session(
        "test",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    state = ScopedStateContext(
        store=store,
        namespace="test",
        session_id=session.id,
        run_id="00000000-0000-4000-8000-000000000002",
    ).for_session("plugin.counter")

    first = await state.append("counter", {"count": 1})

    assert (await state.read("counter")) == [first]
    assert (
        await ScopedStateContext(
            store=store,
            namespace="test",
            session_id=session.id,
            run_id="00000000-0000-4000-8000-000000000002",
        )
        .for_session("plugin.other")
        .read("counter")
        == []
    )
    with raises(StateSequenceConflictError):
        await state.append("counter", {"count": 2})


async def test_runtime_rejects_a_session_pin_mismatch_before_model_execution() -> None:
    calls = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "must not run"

    first = AgentSpec(name="first", model="test:model")
    second = AgentSpec(name="second", model="test:model")
    store = MemoryStore()
    runtime = Runtime(store=store)
    session = await runtime.create_session("test", spec=first, model_id="test:model")

    with raises(SessionPinMismatchError, match="different agent configuration"):
        await runtime.run(
            "test",
            session.id,
            "Do not execute",
            spec=second,
            model=FunctionModel(stream_function=respond),
            deps=None,
            deps_type=type(None),
            idempotency_key="mismatch",
        )

    assert calls == 0
    assert store.run is None


async def test_runtime_cancellation_has_one_terminal_result() -> None:
    spec = AgentSpec(model="test:model")
    store = MemoryStore()
    runtime = Runtime(store=store)
    session = await runtime.create_session("test", spec=spec, model_id="test:model")
    await store.reserve_run(
        "test",
        session.id,
        idempotency_key="cancel",
        prompt="Cancel me",
    )

    assert store.run is not None
    run_id = store.run.id
    first = await runtime.cancel_run("test", run_id)
    second = await runtime.cancel_run("test", run_id)

    assert first.status is RunStatus.INTERRUPTED
    assert second == first
    checked_out = await runtime.checkout(
        "test",
        session.id,
        run_id,
        expected_revision="cancelled",
    )
    assert checked_out.head_run_id == first.id


async def test_runtime_creates_a_session_and_completes_one_run() -> None:
    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "runtime works"

    spec = AgentSpec(model="test:model")
    runtime = Runtime(store=MemoryStore())
    session = await runtime.create_session("test", spec=spec, model_id="test:model")
    sink = EventSink([])

    completion = await runtime.run(
        "test",
        session.id,
        "Say hello",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        idempotency_key="first",
        event_sink=sink,
    )

    assert completion.run.status is RunStatus.COMPLETED
    assert sink.events
    assert completion.run.output == "runtime works"
    final_part = completion.session.messages[-1].parts[-1]
    assert isinstance(final_part, TextPart)
    assert final_part.content == "runtime works"


async def test_runtime_keeps_canonical_history_when_compaction_rewrites_active_history() -> None:
    from yolop_context import Compaction

    class EntryPoint:
        name = "Compaction"
        value = "yolop_context:Compaction"
        dist = None

        @staticmethod
        def load() -> type[Compaction]:
            return Compaction

    spec = AgentSpec(
        model="test:model",
        capabilities=[
            {
                "Compaction": {
                    "target_tokens": 1,
                    "keep_tool_pairs": 0,
                    "include_summarizer": False,
                }
            }
        ],
    )
    prior = [
        ModelResponse(parts=[ToolCallPart("read_file", {}, tool_call_id="call")]),
        ModelRequest(
            parts=[ToolReturnPart("read_file", "old canonical result", tool_call_id="call")]
        ),
    ]
    store = MemoryStore()
    catalog = ProviderCatalog.from_entry_points(capability_entry_points=[EntryPoint()])
    runtime = Runtime(store=store, provider_catalog=catalog)
    session = await runtime.create_session("test", spec=spec, model_id="test:model")
    assert store.session is not None
    store.session = replace(store.session, messages=prior, revision="prior")

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "new answer"

    completion = await runtime.run(
        "test",
        session.id,
        "current prompt",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        idempotency_key="compaction",
    )

    assert any(
        isinstance(message, ModelRequest)
        and any(
            isinstance(part, ToolReturnPart) and part.content == "old canonical result"
            for part in message.parts
        )
        for message in completion.run.full_messages
    )
    assert any(
        isinstance(message, ModelRequest)
        and any(
            isinstance(part, ToolReturnPart) and part.content == "[tool result cleared]"
            for part in message.parts
        )
        for message in completion.run.active_messages
    )
    assert any(
        isinstance(part, TextPart) and part.content == "new answer"
        for message in completion.run.full_messages
        if isinstance(message, ModelResponse)
        for part in message.parts
    )
