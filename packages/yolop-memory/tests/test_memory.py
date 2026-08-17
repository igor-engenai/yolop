from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from pydantic_ai import AgentRunResultEvent, AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pytest import raises
from yolop_memory import (
    MemoryDestructiveOperationDeniedError,
    MemoryHostPolicy,
    MemoryLimits,
    MemoryRevisionConflictError,
    MemoryScope,
    MemoryScopeForbiddenError,
    MemoryScopeKind,
    MemoryToolForbiddenError,
    RuntimeMemoryStore,
    build_memory_capability,
)
from yolop_runtime import ExecutionPin, ExecutionScope
from yolop_sqlite_session import SQLiteRuntimeStore

from yolop import Yolop


@dataclass
class HostDeps:
    scope: ExecutionScope


async def test_native_memory_capability_writes_with_execution_provenance(tmp_path) -> None:
    runtime_store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session_id = str(uuid4())
    run_id = str(uuid4())
    memory = RuntimeMemoryStore(
        runtime_store,
        namespace="tenant/acme",
        scope=MemoryScope.session(session_id),
    )
    spec = AgentSpec(
        metadata={"memory": {"scopes": ["session"], "tools": ["write"]}},
    )
    capability = build_memory_capability(
        spec,
        store_for_scope=lambda _ctx, _scope: memory,
    )
    assert capability is not None
    deps = HostDeps(
        scope=ExecutionScope(
            namespace="tenant/acme",
            session_id=session_id,
            run_id=run_id,
        )
    )

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            yield {
                0: DeltaToolCall(
                    name="memory_write",
                    json_args='{"scope":"session","content":"Use uv","provenance":"agent"}',
                    tool_call_id="memory-write",
                )
            }
        else:
            yield "memory saved"

    async with Yolop().run(
        spec,
        "save this preference",
        model=FunctionModel(stream_function=respond),
        deps=deps,
        deps_type=HostDeps,
        mandatory_capabilities=[capability],
    ) as run:
        events = [event async for event in run]

    records = await memory.search("uv")
    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "memory saved"
    assert records[0].created_by_run_id == run_id


def test_agent_memory_selection_cannot_expand_host_policy() -> None:
    spec = AgentSpec(
        metadata={"memory": {"scopes": ["workspace"], "tools": ["retire"]}},
    )
    policy = MemoryHostPolicy(
        allowed_scopes=frozenset({MemoryScopeKind.SESSION}),
        allowed_tools=frozenset({"read"}),
    )

    with raises(MemoryScopeForbiddenError):
        build_memory_capability(
            spec,
            store_for_scope=cast(Any, lambda _ctx, _scope: None),
            host_policy=policy,
        )

    tool_spec = AgentSpec(metadata={"memory": {"scopes": ["session"], "tools": ["retire"]}})
    with raises(MemoryToolForbiddenError):
        build_memory_capability(
            tool_spec,
            store_for_scope=cast(Any, lambda _ctx, _scope: None),
            host_policy=policy,
        )


async def test_memory_writes_and_reads_through_runtime_store(tmp_path) -> None:
    runtime_store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await runtime_store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    memory = RuntimeMemoryStore(
        runtime_store,
        namespace="tenant/acme",
        scope=MemoryScope.session(session.id),
    )

    created = await memory.create(
        "Use uv for Python commands.",
        created_by_run_id=session.id,
        provenance="test",
    )

    found = await memory.get(created.memory_id)

    assert found == created
    assert found.content == "Use uv for Python commands."


async def test_user_scope_cross_session_and_namespace_isolation(tmp_path) -> None:
    runtime_store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    first = await runtime_store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    second = await runtime_store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    beta = await runtime_store.create_session(
        "tenant/beta",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    scope = MemoryScope.user("user-1")
    first_memory = RuntimeMemoryStore(runtime_store, namespace="tenant/acme", scope=scope)
    second_memory = RuntimeMemoryStore(runtime_store, namespace="tenant/acme", scope=scope)
    other_user = RuntimeMemoryStore(
        runtime_store,
        namespace="tenant/acme",
        scope=MemoryScope.user("user-2"),
    )
    other_namespace = RuntimeMemoryStore(runtime_store, namespace="tenant/beta", scope=scope)

    created = await first_memory.create(
        "shared preference",
        created_by_run_id=first.id,
        provenance="test",
    )

    assert (await second_memory.get(created.memory_id)) == created
    assert await other_user.get(created.memory_id) is None
    assert await other_namespace.get(created.memory_id) is None
    assert beta.id != second.id


async def test_replace_uses_revision_cas_and_preserves_superseded_history(tmp_path) -> None:
    runtime_store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await runtime_store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    memory = RuntimeMemoryStore(
        runtime_store,
        namespace="tenant/acme",
        scope=MemoryScope.session(session.id),
    )
    created = await memory.create(
        "old content",
        created_by_run_id=session.id,
        provenance="initial",
    )

    replacement = await memory.replace(
        created.memory_id,
        expected_revision=1,
        content="new content",
        updated_by_run_id=session.id,
        provenance="updated",
    )

    with raises(MemoryRevisionConflictError):
        await memory.replace(
            created.memory_id,
            expected_revision=1,
            content="stale",
            updated_by_run_id=session.id,
            provenance="stale",
        )
    history = await memory.history(created.memory_id)
    assert [item.status for item in history] == ["superseded", "active"]
    assert history[0].content == "old content"
    assert replacement.created_by_run_id == session.id
    assert replacement.provenance == "updated"


async def test_search_is_bounded_and_survives_store_restart(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    runtime_store = SQLiteRuntimeStore(database)
    session = await runtime_store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    limits = MemoryLimits(max_results=1, max_result_bytes=512)
    memory = RuntimeMemoryStore(
        runtime_store,
        namespace="tenant/acme",
        scope=MemoryScope.session(session.id),
        limits=limits,
    )
    await memory.create(
        "python tooling preference",
        created_by_run_id=session.id,
        provenance="test",
    )
    await memory.create(
        "python testing preference",
        created_by_run_id=session.id,
        provenance="test",
    )

    first = await memory.search("python", limit=20)
    reopened = RuntimeMemoryStore(
        SQLiteRuntimeStore(database),
        namespace="tenant/acme",
        scope=MemoryScope.session(session.id),
        limits=limits,
    )
    second = await reopened.search("python", limit=20)

    assert len(first) == 1
    assert second == first


async def test_retire_requires_host_policy_and_preserves_audit_history(tmp_path) -> None:
    runtime_store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await runtime_store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    memory = RuntimeMemoryStore(
        runtime_store,
        namespace="tenant/acme",
        scope=MemoryScope.session(session.id),
    )
    created = await memory.create(
        "retire me",
        created_by_run_id=session.id,
        provenance="test",
    )

    with raises(MemoryDestructiveOperationDeniedError):
        await memory.retire(
            created.memory_id,
            expected_revision=1,
            retired_by_run_id=session.id,
            provenance="blocked",
        )

    allowed = RuntimeMemoryStore(
        runtime_store,
        namespace="tenant/acme",
        scope=MemoryScope.session(session.id),
        limits=MemoryLimits(allow_retire=True),
    )
    retired = await allowed.retire(
        created.memory_id,
        expected_revision=1,
        retired_by_run_id=session.id,
        provenance="reviewed",
    )
    assert retired.status == "retired"
    assert await allowed.get(created.memory_id) is None
    assert (await allowed.history(created.memory_id))[-1].provenance == "reviewed"
