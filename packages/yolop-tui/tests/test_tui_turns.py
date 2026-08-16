import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import AgentSpec
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from yolop_session import ExecutionPin
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_tui import run_tui


class CapturingOutput(DummyOutput):
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, data: str) -> None:
        self.writes.append(data)

    def write_raw(self, data: str) -> None:
        self.writes.append(data)

    def get_size(self) -> Size:
        return Size(rows=24, columns=80)

    @property
    def text(self) -> str:
        return "".join(self.writes)


async def _wait_for_output(output: CapturingOutput, text: str) -> None:
    async with asyncio.timeout(1):
        while text not in output.text:
            await asyncio.sleep(0)


async def test_tui_streams_a_turn_and_saves_exact_session_history(tmp_path: Path) -> None:
    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "Hello from "
        yield "the model"

    output = CapturingOutput()
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            running = asyncio.create_task(
                run_tui(
                    AgentSpec(),
                    store=store,
                    namespace="test",
                    deps=None,
                    deps_type=type(None),
                    model=FunctionModel(stream_function=respond),
                    model_id="test:model",
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("Hello\r")
            await _wait_for_output(output, "Hello from the model")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    session_ids = await store.list_sessions("test")
    assert len(session_ids) == 1
    session = await store.load_session("test", session_ids[0])
    request = session.messages[0]
    response = session.messages[-1]
    assert isinstance(request, ModelRequest)
    assert isinstance(request.parts[-1], UserPromptPart)
    assert request.parts[-1].content == "Hello"
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[-1], TextPart)
    assert response.parts[-1].content == "Hello from the model"


async def test_tui_renders_and_continues_an_existing_session(tmp_path: Path) -> None:
    spec = AgentSpec()
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session(
        "test",
        pin=ExecutionPin.from_spec(spec, model_id="test:model"),
    )
    session = await store.replace_session(
        "test",
        session.id,
        expected_revision=session.revision,
        messages=[
            ModelRequest(parts=[UserPromptPart("Earlier question")]),
            ModelResponse(parts=[TextPart("Earlier answer")]),
        ],
    )

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        assert messages[:2] == session.messages
        yield "Continued answer"

    output = CapturingOutput()
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            running = asyncio.create_task(
                run_tui(
                    spec,
                    store=store,
                    namespace="test",
                    deps=None,
                    deps_type=type(None),
                    model=FunctionModel(stream_function=respond),
                    model_id="test:model",
                    session_id=session.id,
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "Earlier answer")
            pipe_input.send_text("Continue\r")
            await _wait_for_output(output, "Continued answer")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    saved = await store.load_session("test", session.id)
    assert len(saved.messages) == 4


async def test_failed_tui_turn_returns_to_editor_without_saving(tmp_path: Path) -> None:
    async def fail(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        raise RuntimeError("provider secret")
        yield "unreachable"

    output = CapturingOutput()
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            running = asyncio.create_task(
                run_tui(
                    AgentSpec(),
                    store=store,
                    namespace="test",
                    deps=None,
                    deps_type=type(None),
                    model=FunctionModel(stream_function=fail),
                    model_id="test:model",
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("Fail safely\r")
            await _wait_for_output(output, "Error: Agent run failed")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    session_id = (await store.list_sessions("test"))[0]
    assert (await store.load_session("test", session_id)).messages == []
    assert "provider secret" not in output.text
