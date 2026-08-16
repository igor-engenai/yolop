import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import yolop_tui.app as tui_app
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import MonkeyPatch
from textual.pilot import Pilot
from textual.widgets import Static
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_tui import run_tui
from yolop_tui.textual_app import TextualTerminal


async def test_textual_tui_streams_and_persists_a_native_turn(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "Textual model answer"

    async def drive(pilot: Pilot) -> None:
        await pilot.pause()
        status = pilot.app.query_one("#status", Static)
        await pilot.press(*"Hello", "enter")
        async with asyncio.timeout(1):
            while "↑0 ↓0" in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press("/", "q", "u", "i", "t", "enter")

    def terminal_factory(**kwargs) -> TextualTerminal:
        return TextualTerminal(**kwargs, auto_pilot=drive)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")

    await run_tui(
        AgentSpec(),
        store=store,
        namespace="test",
        deps=None,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:model",
        cwd=tmp_path,
    )

    session_id = (await store.list_sessions("test"))[0]
    session = await store.load_session("test", session_id)
    request, response = session.messages
    assert isinstance(request, ModelRequest)
    assert isinstance(request.parts[-1], UserPromptPart)
    assert request.parts[-1].content == "Hello"
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[-1], TextPart)
    assert response.parts[-1].content == "Textual model answer"


async def test_textual_ctrl_c_cancels_and_saves_partial_history(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    model_stopped = asyncio.Event()
    wait_forever = asyncio.Event()

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        try:
            yield "Partial Textual answer"
            await wait_forever.wait()
        finally:
            model_stopped.set()

    async def drive(pilot: Pilot) -> None:
        status = pilot.app.query_one("#status", Static)
        await pilot.press(*"Cancel me", "enter")
        async with asyncio.timeout(1):
            while "running" not in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press("ctrl+c")
        await asyncio.wait_for(model_stopped.wait(), timeout=1)
        async with asyncio.timeout(1):
            while "idle" not in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press("/", "q", "u", "i", "t", "enter")

    def terminal_factory(**kwargs) -> TextualTerminal:
        return TextualTerminal(**kwargs, auto_pilot=drive)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")

    await run_tui(
        AgentSpec(),
        store=store,
        namespace="test",
        deps=None,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:model",
        cwd=tmp_path,
    )

    session_id = (await store.list_sessions("test"))[0]
    session = await store.load_session("test", session_id)
    request, response = session.messages
    assert isinstance(request, ModelRequest)
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[-1], TextPart)
    assert response.parts[-1].content == "Partial Textual answer"


async def test_textual_editor_steers_an_active_native_run(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    release_first = asyncio.Event()
    calls = 0

    async def respond(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            yield "Initial Textual work"
            await release_first.wait()
        else:
            prompts = [
                part.content
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, UserPromptPart)
            ]
            assert prompts[-1] == "Change direction"
            yield "Steered Textual answer"

    async def drive(pilot: Pilot) -> None:
        status = pilot.app.query_one("#status", Static)
        await pilot.press(*"Start", "enter")
        async with asyncio.timeout(1):
            while "running" not in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press(*"Change direction", "enter")
        async with asyncio.timeout(1):
            while "queued 1" not in str(status.render()):
                await pilot.pause(0.01)
        release_first.set()
        async with asyncio.timeout(1):
            while "idle" not in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press("/", "q", "u", "i", "t", "enter")

    def terminal_factory(**kwargs) -> TextualTerminal:
        return TextualTerminal(**kwargs, auto_pilot=drive)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")

    await run_tui(
        AgentSpec(),
        store=store,
        namespace="test",
        deps=None,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:model",
        cwd=tmp_path,
    )

    assert calls == 2
