from collections.abc import AsyncIterator

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import raises
from yolop_runtime import RunStatus, Runtime, SessionConflictError
from yolop_sqlite_session import SQLiteRuntimeStore


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
