import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic_ai import AgentSpec
from pydantic_ai.usage import RunUsage, UsageLimits
from pytest import raises
from yolop_runtime import (
    ExecutionPin,
    PluginStateEntry,
    Runtime,
    ScopedStateContext,
    StateFormatError,
    StatePayloadLimitError,
    StateSchemaError,
    StateScope,
    StateSequenceConflictError,
)
from yolop_sqlite_session import SQLiteRuntimeStore

from yolop import Yolop

SESSION_ID = "00000000-0000-4000-8000-000000000001"


@dataclass
class FakeResult:
    output: Any = "done"
    usage: RunUsage = field(default_factory=RunUsage)

    def new_messages(self):
        return []

    def all_messages(self):
        return []


class CaptureKernel(Yolop):
    async def execute(self, spec, prompt, **kwargs):
        del spec, prompt
        self.state = kwargs["deps"].state
        self.usage_limits = kwargs["usage_limits"]
        return FakeResult()


def test_plugin_state_entry_preserves_opaque_json_and_identity() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)

    entry = PluginStateEntry(
        namespace="tenant/acme",
        owner_id="plugin.counter",
        scope=StateScope.SESSION,
        scope_id=SESSION_ID,
        state_kind="counter",
        schema_version=1,
        sequence=1,
        payload={"count": 1, "plugin_data": ["opaque"]},
        created_at=created_at,
    )

    assert entry.scope is StateScope.SESSION
    assert entry.payload == {"count": 1, "plugin_data": ["opaque"]}
    assert entry.created_at == created_at


def test_plugin_state_entry_rejects_oversized_payload() -> None:
    with raises(StatePayloadLimitError):
        PluginStateEntry(
            namespace="tenant/acme",
            owner_id="plugin.counter",
            scope=StateScope.SESSION,
            scope_id=SESSION_ID,
            state_kind="counter",
            schema_version=1,
            sequence=1,
            payload={"value": "x" * 70_000},
            created_at=datetime.now(UTC),
        )


async def test_runtime_passes_native_usage_limits_to_kernel(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    kernel = CaptureKernel()
    runtime = Runtime(store=store, kernel=kernel)
    spec = AgentSpec(model="test:model")
    session = await runtime.create_session(
        "tenant/acme",
        spec=spec,
        model_id="test:model",
    )
    limits = UsageLimits(request_limit=2, total_tokens_limit=100)

    await runtime.run(
        "tenant/acme",
        session.id,
        "limited",
        spec=spec,
        deps=None,
        deps_type=type(None),
        idempotency_key="limited",
        usage_limits=limits,
    )

    assert kernel.usage_limits is limits


async def test_runtime_deps_expose_state_bound_to_execution_scope(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    kernel = CaptureKernel()
    runtime = Runtime(store=store, kernel=kernel)
    spec = AgentSpec(model="test:model")
    session = await runtime.create_session(
        "tenant/acme",
        spec=spec,
        model_id="test:model",
    )

    completion = await runtime.run(
        "tenant/acme",
        session.id,
        "record state",
        spec=spec,
        deps=None,
        deps_type=type(None),
        idempotency_key="state",
    )

    state = kernel.state.for_session("plugin.counter")
    entry = await state.append("counter", {"run": completion.run.id})
    assert entry.scope is StateScope.SESSION
    assert entry.scope_id == session.id
    assert entry.owner_id == "plugin.counter"


async def test_scoped_state_appends_and_reconstructs_a_session_stream(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    state = ScopedStateContext(
        store=store,
        namespace="tenant/acme",
        session_id=session.id,
        run_id="00000000-0000-4000-8000-000000000002",
    ).for_session("plugin.counter")

    first = await state.append("counter", {"count": 1})
    second = await state.append("counter", {"count": 2}, expected_sequence=first.sequence)

    assert [entry.sequence for entry in await state.read("counter")] == [1, 2]
    assert second.payload == {"count": 2}
    reopened = SQLiteRuntimeStore(tmp_path / "runtime.db")
    reopened_state = ScopedStateContext(
        store=reopened,
        namespace="tenant/acme",
        session_id=session.id,
        run_id="00000000-0000-4000-8000-000000000002",
    ).for_session("plugin.counter")
    assert await reopened_state.read("counter") == await state.read("counter")


async def test_scoped_state_append_uses_compare_and_append_sequence(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    state = ScopedStateContext(
        store=store,
        namespace="tenant/acme",
        session_id=session.id,
        run_id="00000000-0000-4000-8000-000000000002",
    ).for_session("plugin.counter")
    await state.append("counter", {"count": 1})

    with raises(StateSequenceConflictError):
        await state.append("counter", {"count": 2})


async def test_scoped_state_is_owner_namespace_and_scope_isolated(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    acme = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    beta = await store.create_session(
        "tenant/beta",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    acme_context = ScopedStateContext(
        store=store,
        namespace="tenant/acme",
        session_id=acme.id,
        run_id="00000000-0000-4000-8000-000000000002",
    )
    await acme_context.for_session("plugin.counter").append("counter", {"count": 1})

    assert await acme_context.for_session("plugin.other").read("counter") == []
    assert (
        await ScopedStateContext(
            store=store,
            namespace="tenant/beta",
            session_id=beta.id,
            run_id="00000000-0000-4000-8000-000000000003",
        )
        .for_session("plugin.counter")
        .read("counter")
        == []
    )
    assert await acme_context.for_run("plugin.counter").read("counter") == []


async def test_session_deletion_removes_scoped_plugin_state(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    state = ScopedStateContext(
        store=store,
        namespace="tenant/acme",
        session_id=session.id,
        run_id="00000000-0000-4000-8000-000000000002",
    ).for_session("plugin.counter")
    await state.append("counter", {"count": 1})
    await store.delete_session(
        "tenant/acme",
        session.id,
        expected_revision=session.revision,
    )

    assert await state.read("counter") == []


async def test_scoped_state_reports_malformed_and_unsupported_entries(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    session = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    state = ScopedStateContext(
        store=store,
        namespace="tenant/acme",
        session_id=session.id,
        run_id="00000000-0000-4000-8000-000000000002",
    ).for_session("plugin.counter")
    await state.append("counter", {"count": 1})

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE runtime_plugin_state SET schema_version = 99
            WHERE namespace = ? AND owner_id = ? AND scope_id = ?
            """,
            ("tenant/acme", "plugin.counter", session.id),
        )
    with raises(StateSchemaError):
        await state.read("counter", schema_version=1)

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE runtime_plugin_state SET schema_version = 1, payload = ?
            WHERE namespace = ? AND owner_id = ? AND scope_id = ?
            """,
            (b"not-json", "tenant/acme", "plugin.counter", session.id),
        )
    with raises(StateFormatError):
        await state.read("counter")


async def test_scoped_state_concurrent_appends_have_one_winner(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    state = ScopedStateContext(
        store=store,
        namespace="tenant/acme",
        session_id=session.id,
        run_id="00000000-0000-4000-8000-000000000002",
    ).for_session("plugin.counter")

    results = await asyncio.gather(
        state.append("counter", {"value": "first"}),
        state.append("counter", {"value": "second"}),
        return_exceptions=True,
    )

    assert sum(isinstance(result, PluginStateEntry) for result in results) == 1
    assert sum(isinstance(result, StateSequenceConflictError) for result in results) == 1
    assert len(await state.read("counter")) == 1


def test_state_errors_have_stable_codes() -> None:
    assert StateSequenceConflictError.code == "state_sequence_conflict"
    assert StateSchemaError.code == "state_schema_error"
