import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yolop_tui.app as tui_app
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
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.toolsets.function import FunctionToolset
from pytest import MonkeyPatch, fixture, mark, raises
from rich.console import Console, RenderableType
from yolop_context import Compaction
from yolop_runtime import ExecutionPin, Runtime, SessionPinMismatchError
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_tui import run_tui
from yolop_tui.selection import SelectionOption

from yolop import ProviderCatalog


async def _unused_response(
    _messages: list[ModelMessage],
    _info: AgentInfo,
) -> AsyncIterator[str]:
    yield "unexpected"


class CapturingOutput:
    def __init__(self, *, rows: int = 24) -> None:
        self.rows = rows
        self.writes: list[str] = []

    def write(self, data: str) -> None:
        self.writes.append(data)

    @property
    def text(self) -> str:
        return "".join(self.writes)


class _ScriptedInput:
    def __init__(self) -> None:
        self.output: CapturingOutput | None = None
        self.terminal: _ScriptedTerminal | None = None

    def bind(self, terminal: "_ScriptedTerminal") -> None:
        self.terminal = terminal

    def send_text(self, text: str) -> None:
        assert self.terminal is not None
        for character in text:
            self.terminal.feed_text(character)

    def send_bytes(self, data: bytes) -> None:
        assert self.terminal is not None
        for value in data:
            if value == 3:
                self.terminal.interrupt()
            elif value == 4:
                self.terminal.eof()
            elif value == 15:
                self.terminal.toggle_tools()
            elif value == 20:
                self.terminal.toggle_thinking()
            elif value == 27:
                self.terminal.cancel()


class _ScriptedTerminal:
    def __init__(
        self,
        pipe: _ScriptedInput,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], bool | None],
        on_toggle_tools: Callable[[], None],
        on_toggle_thinking: Callable[[], None],
        **_kwargs: Any,
    ) -> None:
        assert pipe.output is not None
        self._pipe = pipe
        self._output = pipe.output
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._on_toggle_tools = on_toggle_tools
        self._on_toggle_thinking = on_toggle_thinking
        self._buffer = ""
        self._stopped = asyncio.Event()
        self._ready = asyncio.Event()
        self._selection: tuple[SelectionOption, ...] | None = None
        self._selection_future: asyncio.Future[str | None] | None = None
        pipe.bind(self)

    async def run(self) -> None:
        self._output.write("╭─ prompt")
        self._ready.set()
        await self._stopped.wait()

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    async def choose(
        self,
        options: list[SelectionOption],
        **_kwargs: Any,
    ) -> str | None:
        self._selection = tuple(options)
        self._selection_future = asyncio.get_running_loop().create_future()
        self._buffer = ""
        self._output.write("\n".join(option.label for option in options))
        try:
            return await self._selection_future
        finally:
            self._selection = None
            self._selection_future = None
            self._buffer = ""

    def set_transcript(self, renderable: RenderableType) -> None:
        stream = StringIO()
        Console(file=stream, width=80, color_system=None).print(renderable)
        self._output.write(stream.getvalue())

    def set_status(self, text: str) -> None:
        self._output.write(text)

    def restore_editor_text(self, text: str) -> None:
        self._buffer = f"{self._buffer}\n\n{text}" if self._buffer else text

    def feed_text(self, character: str) -> None:
        if character == "\r":
            if self._selection is not None:
                matches = [
                    option
                    for option in self._selection
                    if _fuzzy_match(self._buffer.casefold(), option.label.casefold())
                ]
                if matches and self._selection_future is not None:
                    self._selection_future.set_result(matches[0].value)
                return
            text = self._buffer
            self._buffer = ""
            if text.strip():
                if text.strip() == "/quit":
                    asyncio.get_running_loop().call_later(0.01, self._on_submit, text)
                else:
                    self._on_submit(text)
        elif character == "\n":
            self._buffer += "\n"
        else:
            self._buffer += character

    def interrupt(self) -> None:
        if self._buffer:
            self._buffer = ""
        elif self._on_cancel() is not True:
            self.stop()

    def eof(self) -> None:
        if not self._buffer:
            self.stop()

    def cancel(self) -> None:
        if self._selection_future is not None:
            self._selection_future.set_result(None)
        else:
            self._on_cancel()

    def toggle_tools(self) -> None:
        self._on_toggle_tools()

    def toggle_thinking(self) -> None:
        self._on_toggle_thinking()

    def stop(self) -> None:
        self._stopped.set()


_CURRENT_PIPE: _ScriptedInput | None = None


@contextmanager
def create_pipe_input() -> Iterator[_ScriptedInput]:
    global _CURRENT_PIPE
    pipe = _ScriptedInput()
    _CURRENT_PIPE = pipe
    try:
        yield pipe
    finally:
        _CURRENT_PIPE = None


@contextmanager
def create_app_session(
    *,
    input: _ScriptedInput,
    output: CapturingOutput,
) -> Iterator[None]:
    input.output = output
    yield


@fixture(autouse=True)
def use_scripted_terminal(monkeypatch: MonkeyPatch) -> None:
    def terminal_factory(**kwargs: Any) -> _ScriptedTerminal:
        assert _CURRENT_PIPE is not None
        return _ScriptedTerminal(_CURRENT_PIPE, **kwargs)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)


def _fuzzy_match(query: str, value: str) -> bool:
    characters = iter(value)
    return all(any(candidate == expected for candidate in characters) for expected in query)


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
            session_id = (await store.list_sessions("test"))[0]
            await _wait_for_output(output, session_id[:8])
            assert "test:model" in output.text
            assert "↑0 ↓0 · idle" in output.text
            pipe_input.send_text("Hello\r")
            await _wait_for_output(output, "Hello from the model")
            async with asyncio.timeout(1):
                while not (await store.load_session("test", session_id)).messages:
                    await asyncio.sleep(0)
            saved_before_exit = await store.load_session("test", session_id)
            response_before_exit = saved_before_exit.messages[-1]
            assert isinstance(response_before_exit, ModelResponse)
            assert response_before_exit.usage.input_tokens
            assert response_before_exit.usage.output_tokens
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


async def test_tui_resumes_long_history_in_a_small_terminal(tmp_path: Path) -> None:
    spec = AgentSpec()
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session(
        "test",
        pin=ExecutionPin.from_spec(spec, model_id="test:model"),
    )
    long_answer = "\n".join(f"history line {index}" for index in range(30))
    await store.replace_session(
        "test",
        session.id,
        expected_revision=session.revision,
        messages=[
            ModelRequest(parts=[UserPromptPart("Earlier question")]),
            ModelResponse(parts=[TextPart(long_answer)]),
        ],
    )

    output = CapturingOutput(rows=8)
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            running = asyncio.create_task(
                run_tui(
                    spec,
                    store=store,
                    namespace="test",
                    deps=None,
                    deps_type=type(None),
                    model="test:model",
                    model_id="test:model",
                    session_id=session.id,
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "history line 29")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert "Window too small" not in output.text
    assert "╭─ prompt" in output.text


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
            await _wait_for_output(output, "queued 1")
            release_first_response.set()
            await _wait_for_output(output, "Steered answer")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert len(request_run_ids) == 2
    assert request_run_ids[0] == request_run_ids[1]


@mark.parametrize("cancel_key", [b"\x1b", b"\x03"], ids=["escape", "ctrl-c"])
async def test_cancel_key_saves_partial_native_history(
    tmp_path: Path,
    cancel_key: bytes,
) -> None:
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
            pipe_input.send_bytes(cancel_key)
            await asyncio.wait_for(model_stopped.wait(), timeout=1)
            assert not running.done()
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
            await _wait_for_output(output, "wait_tool")
            pipe_input.send_bytes(b"\x1b")
            await asyncio.wait_for(deps.stopped.wait(), timeout=1)
            session_id = (await store.list_sessions("test"))[0]
            async with asyncio.timeout(1):
                while not (await store.load_session("test", session_id)).messages:
                    await asyncio.sleep(0)
            pipe_input.send_text("Continue\r")
            await _wait_for_output(output, "Tool finished")
            detail = "The tool call was interrupted before a result was produced."
            assert detail not in output.text
            pipe_input.send_bytes(b"\x0f")
            await _wait_for_output(output, detail)
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


async def test_help_lists_only_the_minimal_commands(tmp_path: Path) -> None:
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
                    model="test:model",
                    model_id="test:model",
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("/help\r")
            await _wait_for_output(
                output,
                "Commands: /new  /resume  /history  /compact [focus]  /goal <condition>",
            )
            await _wait_for_output(output, "/goal-status")
            await _wait_for_output(output, "Scroll: PageUp/PageDown or mouse wheel")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)


async def test_compact_command_uses_the_selected_runtime_capability(tmp_path: Path) -> None:
    class EntryPoint:
        name = "Compaction"
        value = "yolop_context:Compaction"
        dist = None

        @staticmethod
        def load() -> type[Compaction]:
            return Compaction

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "unexpected"

    spec = AgentSpec(
        model="test:model",
        capabilities=[{"Compaction": {"target_tokens": 1, "include_summarizer": False}}],
    )
    output = CapturingOutput()
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    catalog = ProviderCatalog.from_entry_points(capability_entry_points=[EntryPoint()])
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
                    provider_catalog=catalog,
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("/compact keep {braces}\r")
            await _wait_for_output(output, "Session context compacted")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)


async def test_history_command_handles_empty_session(tmp_path: Path) -> None:
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
                    model="test:model",
                    model_id="test:model",
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("/history\r")
            await _wait_for_output(
                output,
                "No terminal Runs in this Session; use /resume to open another saved Session",
            )
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)


async def test_history_picker_can_fork_a_terminal_run(tmp_path: Path) -> None:
    output = CapturingOutput()
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    spec = AgentSpec(model="test:model")
    runtime = Runtime(store=store)
    session = await runtime.create_session("test", spec=spec, model_id="test:model")
    first = await runtime.run(
        "test",
        session.id,
        "first",
        spec=spec,
        model=FunctionModel(stream_function=_unused_response),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="first",
    )
    await runtime.run_related(
        "test",
        session.id,
        "second",
        parent_run_id=first.run.id,
        spec=spec,
        model=FunctionModel(stream_function=_unused_response),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="second",
    )

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            running = asyncio.create_task(
                run_tui(
                    spec,
                    store=store,
                    namespace="test",
                    deps=None,
                    deps_type=type(None),
                    model=FunctionModel(stream_function=_unused_response),
                    model_id="test:model",
                    session_id=session.id,
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("/history\r")
            await _wait_for_output(output, "Fork ·")
            pipe_input.send_text("Fork\r")
            await _wait_for_output(output, "Forked Session")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert len(await store.list_sessions("test")) == 2


async def test_goal_status_reports_when_no_goal_is_selected(tmp_path: Path) -> None:
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
                    model="test:model",
                    model_id="test:model",
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("/goal-status\r")
            await _wait_for_output(output, "Goal command failed: No goal is selected")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)


async def test_login_explains_how_to_install_auth_providers(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(tui_app, "_AUTH_PROVIDER_LOADER", lambda: ())
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
                    model="test:model",
                    model_id="test:model",
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("/login\r")
            await _wait_for_output(output, "No authentication providers are installed")
            assert "yolop[tui,providers]" in output.text
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)


async def test_new_command_switches_to_a_fresh_durable_session(tmp_path: Path) -> None:
    calls = 0

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "unexpected"

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
            first_id = (await store.list_sessions("test"))[0]
            pipe_input.send_text("/new\r")
            async with asyncio.timeout(1):
                while len(await store.list_sessions("test")) < 2:
                    await asyncio.sleep(0)
            second_id = next(
                session_id
                for session_id in await store.list_sessions("test")
                if session_id != first_id
            )
            await _wait_for_output(output, second_id[:8])
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert calls == 0


async def test_session_command_is_rejected_while_a_run_is_active(tmp_path: Path) -> None:
    model_stopped = asyncio.Event()
    wait_forever = asyncio.Event()

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        try:
            yield "Working"
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
            pipe_input.send_text("Start\r")
            await _wait_for_output(output, "Working")
            pipe_input.send_text("/new\r")
            await _wait_for_output(output, "Cancel the active run before changing sessions")
            assert len(await store.list_sessions("test")) == 1
            pipe_input.send_bytes(b"\x1b")
            await asyncio.wait_for(model_stopped.wait(), timeout=1)
            session_id = (await store.list_sessions("test"))[0]
            async with asyncio.timeout(1):
                while not (await store.load_session("test", session_id)).messages:
                    await asyncio.sleep(0)
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)


async def test_resume_command_fuzzy_selects_a_pinned_session(tmp_path: Path) -> None:
    calls = 0

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "unexpected"

    spec = AgentSpec()
    pin = ExecutionPin.from_spec(spec, model_id="test:model")
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    target = await store.create_session("test", pin=pin)
    target = await store.replace_session(
        "test",
        target.id,
        expected_revision=target.revision,
        messages=[
            ModelRequest(parts=[UserPromptPart("Target conversation")]),
            ModelResponse(parts=[TextPart("Existing answer")]),
        ],
    )
    other_spec = AgentSpec(instructions="other")
    other = await store.create_session(
        "test",
        pin=ExecutionPin.from_spec(other_spec, model_id="test:model"),
    )

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
                    cwd=tmp_path,
                )
            )
            await _wait_for_output(output, "╭─ prompt")
            pipe_input.send_text("/resume\r")
            await _wait_for_output(output, "Target conversation")
            await _wait_for_output(output, "(empty session)")
            assert other.id not in output.text
            pipe_input.send_text("Target\r")
            await _wait_for_output(output, target.id[:8])
            await _wait_for_output(output, "Existing answer")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert calls == 0


async def test_tui_rejects_a_session_pinned_to_another_agent(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    other = await store.create_session(
        "test",
        pin=ExecutionPin.from_spec(AgentSpec(instructions="other"), model_id="test:model"),
    )

    with raises(SessionPinMismatchError, match="pinned to different agent configuration"):
        await run_tui(
            AgentSpec(),
            store=store,
            namespace="test",
            deps=None,
            deps_type=type(None),
            model="test:model",
            model_id="test:model",
            session_id=other.id,
            cwd=tmp_path,
        )


async def test_thinking_is_hidden_until_ctrl_t(tmp_path: Path) -> None:
    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str | DeltaThinkingCalls]:
        yield {0: DeltaThinkingPart(content="private thought")}
        yield "Visible answer"

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
            pipe_input.send_text("Think\r")
            await _wait_for_output(output, "Visible answer")
            assert "private thought" not in output.text
            pipe_input.send_bytes(b"\x14")
            await _wait_for_output(output, "private thought")
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)
