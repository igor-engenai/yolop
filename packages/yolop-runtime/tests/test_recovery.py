from collections.abc import AsyncIterator
from inspect import signature

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, TextContent, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import mark, raises
from yolop_runtime import RunStateError, RunStatus, Runtime
from yolop_sqlite_session import SQLiteRuntimeStore


def test_runtime_exposes_explicit_interrupted_run_retry() -> None:
    assert "retry_interrupted_run" in dir(Runtime)
    assert "idempotency_key" in signature(Runtime.retry_interrupted_run).parameters


@mark.parametrize("prompt", ["start", [TextContent("start")]])
async def test_interrupted_run_requires_explicit_idempotent_retry(tmp_path, prompt) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    spec = AgentSpec(model="test:model")
    session = await runtime.create_session("test", spec=spec, model_id="test:model")
    reservation = await runtime.reserve_run(
        "test",
        session.id,
        prompt,
        spec=spec,
        model_id="test:model",
        idempotency_key="request",
    )
    claimed = await store.claim_run(
        "test",
        reservation.run.id,
        owner_id="worker-1",
        lease_seconds=30,
    )
    assert await store.interrupt_owned_runs("worker-1") == 1
    assert (await store.load_run("test", claimed.id)).status is RunStatus.INTERRUPTED

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str]:
        assert any(
            isinstance(part, UserPromptPart) and part.content == prompt
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        yield "retried"

    resumed = await runtime.retry_interrupted_run(
        "test",
        session.id,
        claimed.id,
        prompt,
        spec=spec,
        model=FunctionModel(stream_function=respond),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="retry",
    )

    assert resumed.run.output == "retried"
    with raises(RunStateError):
        await runtime.retry_interrupted_run(
            "test",
            session.id,
            resumed.run.id,
            prompt,
            spec=spec,
            model=FunctionModel(stream_function=respond),
            model_id="test:model",
            deps=None,
            deps_type=type(None),
            idempotency_key="retry-again",
        )
