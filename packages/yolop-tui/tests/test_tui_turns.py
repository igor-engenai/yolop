import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import AgentSpec, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.toolsets.function import FunctionToolset
from pytest import MonkeyPatch
from yolop_session import ExecutionPin
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_tui import run_tui


@dataclass
class ToolDeps:
    started: asyncio.Event
    stopped: asyncio.Event
    wait_forever: asyncio.Event


@dataclass
class WaitingTool(AbstractCapability[ToolDeps]):
    _toolset: FunctionToolset[ToolDeps] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._toolset = FunctionToolset[ToolDeps](
            [Tool(self.wait_tool, takes_ctx=True)],
            id="waiting-tool",
        )

    def get_toolset(self) -> FunctionToolset[ToolDeps]:
        return self._toolset

    async def wait_tool(self, context: RunContext[ToolDeps]) -> str:
        context.deps.started.set()
        try:
            await context.deps.wait_forever.wait()
        finally:
            context.deps.stopped.set()
        return "finished"


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


async def test_input_during_a_run_steers_the_same_native_agent_run(tmp_path: Path) -> None:
    release_first_response = asyncio.Event()
    request_run_ids: list[str | None] = []

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        request_run_ids.append(request.run_id)
        if len(request_run_ids) == 1:
            yield "Initial work"
            await release_first_response.wait()
        else:
            user_prompt = request.parts[-1]
            assert isinstance(user_prompt, UserPromptPart)
            assert user_prompt.content == "Change direction"
            yield "Steered answer"

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
            pipe_input.send_text("Start\r")
            await _wait_for_output(output, "Initial work")
            pipe_input.send_text("Change direction\r")
            await asyncio.sleep(0)
            release_first_response.set()
            await _wait_for_output(output, "Steered answer")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert len(request_run_ids) == 2
    assert request_run_ids[0] == request_run_ids[1]


async def test_escape_cancels_and_saves_partial_native_history(tmp_path: Path) -> None:
    model_stopped = asyncio.Event()
    wait_forever = asyncio.Event()

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        try:
            yield "Partial answer"
            await wait_forever.wait()
        finally:
            model_stopped.set()

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
            pipe_input.send_text("Start and stop\r")
            await _wait_for_output(output, "Partial answer")
            pipe_input.send_bytes(b"\x1b")
            await asyncio.wait_for(model_stopped.wait(), timeout=1)
            session_id = (await store.list_sessions("test"))[0]
            async with asyncio.timeout(1):
                while not (await store.load_session("test", session_id)).messages:
                    await asyncio.sleep(0)
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    session = await store.load_session("test", session_id)
    request, response = session.messages
    assert isinstance(request, ModelRequest)
    assert isinstance(request.parts[-1], UserPromptPart)
    assert request.parts[-1].content == "Start and stop"
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[-1], TextPart)
    assert response.parts[-1].content == "Partial answer"


async def test_multiple_steering_messages_are_delivered_in_order(tmp_path: Path) -> None:
    release_first_response = asyncio.Event()
    calls = 0

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            yield "Working"
            await release_first_response.wait()
        else:
            prompts = [
                part.content
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, UserPromptPart)
            ]
            assert prompts[-2:] == ["First change", "Second change"]
            yield "Both changes applied"

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
            pipe_input.send_text("Start\r")
            await _wait_for_output(output, "Working")
            pipe_input.send_text("First change\rSecond change\r")
            await asyncio.sleep(0)
            release_first_response.set()
            await _wait_for_output(output, "Both changes applied")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert calls == 2


async def test_cancel_restores_undelivered_steering_text_to_editor(tmp_path: Path) -> None:
    model_stopped = asyncio.Event()
    wait_forever = asyncio.Event()
    calls = 0

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                yield "Holding"
                await wait_forever.wait()
            finally:
                model_stopped.set()
        else:
            request = messages[-1]
            assert isinstance(request, ModelRequest)
            prompt = request.parts[-1]
            assert isinstance(prompt, UserPromptPart)
            assert prompt.content == "Retry me"
            yield "Restored prompt ran"

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
            pipe_input.send_text("Start\r")
            await _wait_for_output(output, "Holding")
            pipe_input.send_text("Retry me\r")
            await asyncio.sleep(0)
            pipe_input.send_bytes(b"\x1b")
            await asyncio.wait_for(model_stopped.wait(), timeout=1)
            session_id = (await store.list_sessions("test"))[0]
            async with asyncio.timeout(1):
                while len((await store.load_session("test", session_id)).messages) < 2:
                    await asyncio.sleep(0)
            pipe_input.send_text("\r")
            await _wait_for_output(output, "Restored prompt ran")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert calls == 2


async def test_escape_cancels_an_active_tool_and_saves_interrupted_return(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    entry_point = SimpleNamespace(
        name="WaitingTool",
        value="tests:WaitingTool",
        load=lambda: WaitingTool,
    )
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: (entry_point,))
    deps = ToolDeps(asyncio.Event(), asyncio.Event(), asyncio.Event())

    async def respond(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            yield {
                0: DeltaToolCall(
                    name="wait_tool",
                    json_args=json.dumps({}),
                    tool_call_id="waiting",
                )
            }
        else:
            yield "Tool finished"

    output = CapturingOutput()
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    spec = AgentSpec.model_validate({"capabilities": ["WaitingTool"]})
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            running = asyncio.create_task(
                run_tui(
                    spec,
                    store=store,
                    namespace="test",
                    deps=deps,
                    deps_type=ToolDeps,
                    model=FunctionModel(stream_function=respond),
                    model_id="test:model",
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("Use the tool\r")
            await asyncio.wait_for(deps.started.wait(), timeout=1)
            pipe_input.send_bytes(b"\x1b")
            await asyncio.wait_for(deps.stopped.wait(), timeout=1)
            session_id = (await store.list_sessions("test"))[0]
            async with asyncio.timeout(1):
                while not (await store.load_session("test", session_id)).messages:
                    await asyncio.sleep(0)
            pipe_input.send_text("Continue\r")
            await _wait_for_output(output, "Tool finished")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    session = await store.load_session("test", session_id)
    interrupted = [
        part
        for message in session.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(interrupted) == 1
    assert interrupted[0].outcome == "interrupted"


async def test_session_conflict_does_not_overwrite_external_history(tmp_path: Path) -> None:
    release_response = asyncio.Event()
    spec = AgentSpec()
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session(
        "test",
        pin=ExecutionPin.from_spec(spec, model_id="test:model"),
    )

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "Local answer"
        await release_response.wait()

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
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("Local prompt\r")
            await _wait_for_output(output, "Local answer")
            external_messages: list[ModelMessage] = [
                ModelRequest(parts=[UserPromptPart("External prompt")]),
                ModelResponse(parts=[TextPart("External answer")]),
            ]
            await store.replace_session(
                "test",
                session.id,
                expected_revision=session.revision,
                messages=external_messages,
            )
            release_response.set()
            await _wait_for_output(output, "Session changed; the local run was not saved")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert (await store.load_session("test", session.id)).messages == external_messages
