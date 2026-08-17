import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import raises
from yolop_runtime import (
    RunBudgetExceededError,
    RunRelation,
    RunStatus,
    Runtime,
    RuntimeBudget,
    RuntimeDeadlineExceededError,
    SessionConflictError,
)
from yolop_sqlite_session import SQLiteRuntimeStore


async def test_runtime_deadline_interrupts_active_work(tmp_path) -> None:
    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        await asyncio.sleep(60)
        yield "unreachable"

    spec = AgentSpec(model="test:model")
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    session = await runtime.create_session("tenant/acme", spec=spec, model_id="test:model")

    with raises(RuntimeDeadlineExceededError):
        await runtime.run(
            "tenant/acme",
            session.id,
            "expire during execution",
            spec=spec,
            model=FunctionModel(stream_function=respond),
            deps=None,
            deps_type=type(None),
            idempotency_key="active-expired",
            root_budget=RuntimeBudget(wall_deadline=datetime.now(UTC) + timedelta(milliseconds=20)),
        )

    runs = await runtime.list_runs("tenant/acme", session_id=session.id)
    assert len(runs) == 1
    assert runs[0].status is RunStatus.INTERRUPTED
    assert runs[0].error_code == "root_deadline"


async def test_runtime_runs_related_work_with_explicit_ancestry(tmp_path) -> None:
    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "answer"

    spec = AgentSpec(model="test:model")
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    session = await runtime.create_session("tenant/acme", spec=spec, model_id="test:model")
    first = await runtime.run(
        "tenant/acme",
        session.id,
        "root",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        idempotency_key="root",
        root_budget=RuntimeBudget(child_run_limit=1),
    )

    child = await runtime.run_related(
        "tenant/acme",
        session.id,
        "child",
        parent_run_id=first.run.id,
        relation=RunRelation.CHILD,
        spec=spec,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        idempotency_key="child",
    )

    assert child.run.parent_run_id == first.run.id
    assert child.run.root_run_id == first.run.id
    assert child.run.relation is RunRelation.CHILD


async def test_runtime_rejects_work_after_root_wall_deadline(tmp_path) -> None:
    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "must not run"

    spec = AgentSpec(model="test:model")
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store, clock=lambda: datetime.now(UTC))
    session = await runtime.create_session("tenant/acme", spec=spec, model_id="test:model")

    with raises(RuntimeDeadlineExceededError):
        await runtime.run(
            "tenant/acme",
            session.id,
            "expired",
            spec=spec,
            model=FunctionModel(stream_function=respond),
            deps=None,
            deps_type=type(None),
            idempotency_key="expired",
            root_budget=RuntimeBudget(wall_deadline=datetime.now(UTC) - timedelta(seconds=1)),
        )

    assert await runtime.list_runs("tenant/acme", session_id=session.id) == []


async def test_runtime_root_budget_blocks_a_continuation_after_usage(tmp_path) -> None:
    calls = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "answer"

    spec = AgentSpec(model="test:model")
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    session = await runtime.create_session("tenant/acme", spec=spec, model_id="test:model")
    budget = RuntimeBudget(request_limit=1, continuation_limit=2)

    await runtime.run(
        "tenant/acme",
        session.id,
        "first",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        idempotency_key="first",
        root_budget=budget,
    )

    with raises(RunBudgetExceededError):
        await runtime.run(
            "tenant/acme",
            session.id,
            "second",
            spec=spec,
            model=FunctionModel(stream_function=respond),
            deps=None,
            deps_type=type(None),
            idempotency_key="second",
        )

    assert calls == 1


async def test_failed_run_does_not_move_the_session_head(tmp_path) -> None:
    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("provider detail")
        yield "unreachable"

    spec = AgentSpec(model="test:model")
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    session = await runtime.create_session("tenant/acme", spec=spec, model_id="test:model")

    with raises(RuntimeError, match="provider detail"):
        await runtime.run(
            "tenant/acme",
            session.id,
            "fail",
            spec=spec,
            model=FunctionModel(stream_function=fail),
            deps=None,
            deps_type=type(None),
            idempotency_key="fail",
        )

    failed = (await runtime.list_runs("tenant/acme", session_id=session.id))[0]
    assert failed.status is RunStatus.FAILED
    assert (await runtime.load_session("tenant/acme", session.id)).head_run_id is None


async def test_runtime_run_history_is_linear_branchable_and_restartable(tmp_path) -> None:
    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "answer"

    spec = AgentSpec(model="test:model")
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    session = await runtime.create_session("tenant/acme", spec=spec, model_id="test:model")

    first = await runtime.run(
        "tenant/acme",
        session.id,
        "first",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        idempotency_key="first",
    )
    second = await runtime.run(
        "tenant/acme",
        session.id,
        "second",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        idempotency_key="second",
    )

    assert first.run.status is RunStatus.COMPLETED
    assert second.run.parent_run_id == first.run.id
    assert second.run.root_run_id == first.run.id
    assert len(second.run.full_messages) == 4
    assert second.session.head_run_id == second.run.id

    checked_out = await runtime.checkout(
        "tenant/acme",
        session.id,
        first.run.id,
        expected_revision=second.session.revision,
    )
    assert checked_out.head_run_id == first.run.id
    assert (await runtime.get_run("tenant/acme", second.run.id)).status is RunStatus.COMPLETED

    with raises(SessionConflictError):
        await runtime.checkout(
            "tenant/acme",
            session.id,
            second.run.id,
            expected_revision=second.session.revision,
        )

    reopened = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    loaded = await reopened.load_session("tenant/acme", session.id)
    assert loaded.head_run_id == first.run.id
    reopened_second = await reopened.get_run("tenant/acme", second.run.id)
    assert reopened_second.full_messages == second.run.full_messages
    assert reopened_second.events
