from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import raises
from test_delegation import definition
from yolop_delegation import (
    BackgroundDelegationService,
    BackgroundTaskConflictError,
    BackgroundTaskNotActiveError,
    BackgroundTaskStatus,
    DelegateCatalog,
)
from yolop_runtime import RunStatus, Runtime
from yolop_sqlite_session import SQLiteRuntimeStore


def catalog() -> DelegateCatalog:
    return DelegateCatalog(
        {
            "tenant/acme": [
                definition(alias="research", model_id="child:model", max_depth=2, max_children=2)
            ]
        }
    )


async def make_parent(runtime: Runtime[None]) -> tuple[AgentSpec, str, str]:
    spec = AgentSpec(
        model="parent:model",
        metadata={"delegation": {"delegates": [{"alias": "research"}]}},
    )
    session = await runtime.create_session("tenant/acme", spec=spec, model_id="parent:model")
    parent = await runtime.store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="parent",
        prompt="Parent",
    )
    return spec, session.id, parent.run.id


async def test_background_start_is_durable_and_idempotent(tmp_path: Path) -> None:
    runtime = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    spec, parent_session_id, parent_run_id = await make_parent(runtime)
    service = BackgroundDelegationService(
        runtime,
        catalog=catalog(),
        model_for_id=lambda model_id: model_id,
        deps_for_request=lambda _request: (None, type(None)),
    )

    first = await service.start(
        "tenant/acme",
        parent_session_id,
        parent_run_id,
        parent_spec=spec,
        alias="research",
        task="investigate",
        operation_key="op-1",
    )
    second = await service.start(
        "tenant/acme",
        parent_session_id,
        parent_run_id,
        parent_spec=spec,
        alias="research",
        task="investigate",
        operation_key="op-1",
    )

    assert first == second
    assert first.status is BackgroundTaskStatus.ACCEPTED
    assert len(await runtime.list_sessions("tenant/acme")) == 2
    assert await service.list_tasks("tenant/acme", parent_session_id) == [first]

    with_background_conflict = service.start(
        "tenant/acme",
        parent_session_id,
        parent_run_id,
        parent_spec=spec,
        alias="research",
        task="different",
        operation_key="op-1",
    )
    try:
        await with_background_conflict
    except BackgroundTaskConflictError:
        pass
    else:
        raise AssertionError("different task must conflict with an existing operation")


async def test_background_worker_is_restartable_and_collect_is_bounded(tmp_path: Path) -> None:
    runtime = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    spec, parent_session_id, parent_run_id = await make_parent(runtime)

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "0123456789"

    child_model = FunctionModel(stream_function=respond)
    service = BackgroundDelegationService(
        runtime,
        catalog=catalog(),
        model_for_id=lambda model_id: child_model if model_id == "child:model" else model_id,
        deps_for_request=lambda _request: (None, type(None)),
        max_output_bytes=4,
    )
    handle = await service.start(
        "tenant/acme",
        parent_session_id,
        parent_run_id,
        parent_spec=spec,
        alias="research",
        task="investigate",
        operation_key="op-2",
    )

    restarted = BackgroundDelegationService(
        runtime,
        catalog=catalog(),
        model_for_id=lambda model_id: child_model if model_id == "child:model" else model_id,
        deps_for_request=lambda _request: (None, type(None)),
        max_output_bytes=4,
    )
    completed = await restarted.run_worker(handle)
    assert completed.status is BackgroundTaskStatus.COMPLETED
    assert completed.run_status is RunStatus.COMPLETED
    assert completed.output == "0123"
    assert completed.output_truncated is True
    assert (await restarted.inspect(handle)).child_run_id == handle.child_run_id


async def test_background_cancel_is_durable(tmp_path: Path) -> None:
    runtime = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    spec, parent_session_id, parent_run_id = await make_parent(runtime)
    service = BackgroundDelegationService(
        runtime,
        catalog=catalog(),
        model_for_id=lambda model_id: model_id,
        deps_for_request=lambda _request: (None, type(None)),
    )
    handle = await service.start(
        "tenant/acme",
        parent_session_id,
        parent_run_id,
        parent_spec=spec,
        alias="research",
        task="investigate",
        operation_key="op-3",
    )

    cancelled = await service.cancel(handle)

    assert cancelled.status is BackgroundTaskStatus.INTERRUPTED
    assert cancelled.run_status is RunStatus.INTERRUPTED


async def test_background_steering_is_ephemeral_and_uses_the_worker_sink(tmp_path: Path) -> None:
    runtime = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    spec, parent_session_id, parent_run_id = await make_parent(runtime)
    service = BackgroundDelegationService(
        runtime,
        catalog=catalog(),
        model_for_id=lambda model_id: model_id,
        deps_for_request=lambda _request: (None, type(None)),
    )
    handle = await service.start(
        "tenant/acme",
        parent_session_id,
        parent_run_id,
        parent_spec=spec,
        alias="research",
        task="investigate",
        operation_key="op-4",
    )
    prompts: list[str] = []

    async def sink(prompt: str) -> None:
        prompts.append(prompt)

    with raises(BackgroundTaskNotActiveError):
        await service.steer(handle, "before worker")
    async with service.steering_sink(handle, sink):
        await service.steer(handle, "continue")
    with raises(BackgroundTaskNotActiveError):
        await service.steer(handle, "after worker")
    assert prompts == ["continue"]
