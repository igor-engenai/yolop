import asyncio
from collections.abc import AsyncIterator
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import yolop_tui.app as tui_app
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import LogCaptureFixture, MonkeyPatch
from textual.pilot import Pilot
from textual.widgets import Static
from yolop_session import ExecutionPin
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


async def test_textual_resume_opens_modal_and_selects_a_session(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "unexpected"

    spec = AgentSpec()
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    target = await store.create_session(
        "test",
        pin=ExecutionPin.from_spec(spec, model_id="test:model"),
    )
    target = await store.replace_session(
        "test",
        target.id,
        expected_revision=target.revision,
        messages=[
            ModelRequest(parts=[UserPromptPart("Target conversation")]),
            ModelResponse(parts=[TextPart("Existing answer")]),
        ],
    )

    async def drive(pilot: Pilot) -> None:
        await pilot.press(*"/resume", "enter")
        async with asyncio.timeout(1):
            while not pilot.app.screen.query("#session-filter"):
                await pilot.pause(0.01)
        await pilot.press("t", "g", "t", "enter")
        status = pilot.app.query_one("#status", Static)
        async with asyncio.timeout(1):
            while target.id[:8] not in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press(*"/quit", "enter")

    def terminal_factory(**kwargs) -> TextualTerminal:
        return TextualTerminal(**kwargs, auto_pilot=drive)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)

    await run_tui(
        spec,
        store=store,
        namespace="test",
        deps=None,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:model",
        cwd=tmp_path,
    )

    assert calls == 0


async def test_textual_login_selects_provider_and_runs_its_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    login_calls = 0
    model_calls = 0
    release_login = asyncio.Event()

    class Provider:
        name = "openai-codex"
        label = "OpenAI Codex (ChatGPT Plus/Pro)"

        async def login(self, notify):
            nonlocal login_calls
            login_calls += 1
            notify(
                SimpleNamespace(
                    verification_uri="https://auth.openai.com/codex/device",
                    user_code="ABCD-EFGH",
                    expires_in=900.0,
                )
            )
            await release_login.wait()
            return SimpleNamespace(authenticated=True, expires_at=2_000_000_000.0)

        def status(self):
            return SimpleNamespace(authenticated=False, expires_at=None)

        def logout(self) -> bool:
            return False

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        nonlocal model_calls
        model_calls += 1
        yield "Codex after login"

    def resolve(model_name: str) -> FunctionModel:
        assert model_name == "gpt-5.6-luna"
        return FunctionModel(stream_function=respond)

    async def drive(pilot: Pilot) -> None:
        status = pilot.app.query_one("#status", Static)
        await pilot.press(*"/login", "enter")
        async with asyncio.timeout(1):
            while not pilot.app.screen.query("#session-options"):
                await pilot.pause(0.01)
        await pilot.press("enter")
        async with asyncio.timeout(1):
            while not pilot.app.screen.query("#auth-code"):
                await pilot.pause(0.01)
        assert "ABCD-EFGH" in str(pilot.app.screen.query_one("#auth-code", Static).render())
        assert "logging in" in str(status.render())
        release_login.set()
        async with asyncio.timeout(1):
            while "idle" not in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press(*"Use Codex", "enter")
        async with asyncio.timeout(1):
            while model_calls == 0 or "idle" not in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press(*"/quit", "enter")

    def terminal_factory(**kwargs) -> TextualTerminal:
        return TextualTerminal(**kwargs, auto_pilot=drive)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)
    monkeypatch.setattr(tui_app, "_AUTH_PROVIDER_LOADER", lambda: (Provider(),))
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda **_kwargs: (SimpleNamespace(name="openai-codex", load=lambda: resolve),),
    )
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")

    await run_tui(
        AgentSpec(model="openai-codex:gpt-5.6-luna"),
        store=store,
        namespace="test",
        deps=None,
        deps_type=type(None),
        cwd=tmp_path,
    )

    assert login_calls == 1
    assert model_calls == 1
    session_id = (await store.list_sessions("test"))[0]
    response = (await store.load_session("test", session_id)).messages[-1]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[-1], TextPart)
    assert response.parts[-1].content == "Codex after login"


async def test_textual_login_failure_is_sanitized_and_restores_prompt(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    failed = asyncio.Event()

    class Provider:
        name = "openai-codex"
        label = "OpenAI Codex"

        async def login(self, notify):
            failed.set()
            raise RuntimeError("provider-secret-must-not-leak")

        def status(self):
            return SimpleNamespace(authenticated=False, expires_at=None)

        def logout(self) -> bool:
            return False

    async def drive(pilot: Pilot) -> None:
        status = pilot.app.query_one("#status", Static)
        await pilot.press(*"/login", "enter")
        async with asyncio.timeout(1):
            while not pilot.app.screen.query("#session-options"):
                await pilot.pause(0.01)
        await pilot.press("enter")
        await asyncio.wait_for(failed.wait(), timeout=1)
        async with asyncio.timeout(1):
            while "idle" not in str(status.render()) or pilot.app.screen.query("#auth-login"):
                await pilot.pause(0.01)
        assert pilot.app.query_one("#editor").has_focus
        assert "Provider login failed for openai-codex" in caplog.text
        assert "provider-secret" not in caplog.text
        await pilot.press(*"/quit", "enter")

    def terminal_factory(**kwargs) -> TextualTerminal:
        return TextualTerminal(**kwargs, auto_pilot=drive)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)
    monkeypatch.setattr(tui_app, "_AUTH_PROVIDER_LOADER", lambda: (Provider(),))

    await run_tui(
        AgentSpec(),
        store=SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace="test",
        deps=None,
        deps_type=type(None),
        model="test:model",
        model_id="test:model",
        cwd=tmp_path,
    )


async def test_textual_logout_selects_provider_and_confirms(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    logout_calls = 0

    class Provider:
        name = "openai-codex"
        label = "OpenAI Codex"

        async def login(self, _notify):
            raise AssertionError("login must not run")

        def status(self):
            return SimpleNamespace(authenticated=True, expires_at=2_000_000_000.0)

        def logout(self) -> bool:
            nonlocal logout_calls
            logout_calls += 1
            return True

    async def drive(pilot: Pilot) -> None:
        await pilot.press(*"/logout", "enter")
        async with asyncio.timeout(1):
            while not pilot.app.screen.query("#session-options"):
                await pilot.pause(0.01)
        await pilot.press("enter")
        async with asyncio.timeout(1):
            while not pilot.app.screen.query("#confirm-message"):
                await pilot.pause(0.01)
        assert "openai-codex" in str(
            pilot.app.screen.query_one("#confirm-message", Static).render()
        )
        await pilot.press("y")
        async with asyncio.timeout(1):
            while logout_calls == 0:
                await pilot.pause(0.01)
        await pilot.press(*"/quit", "enter")

    def terminal_factory(**kwargs) -> TextualTerminal:
        return TextualTerminal(**kwargs, auto_pilot=drive)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)
    monkeypatch.setattr(tui_app, "_AUTH_PROVIDER_LOADER", lambda: (Provider(),))

    await run_tui(
        AgentSpec(),
        store=SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace="test",
        deps=None,
        deps_type=type(None),
        model="test:model",
        model_id="test:model",
        cwd=tmp_path,
    )

    assert logout_calls == 1


async def test_textual_login_cancel_restores_an_usable_prompt(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    login_cancelled = asyncio.Event()

    class Provider:
        name = "openai-codex"
        label = "OpenAI Codex"

        async def login(self, notify):
            notify(
                SimpleNamespace(
                    verification_uri="https://auth.openai.com/codex/device",
                    user_code="CODE",
                    expires_in=900.0,
                )
            )
            try:
                await asyncio.Event().wait()
            finally:
                login_cancelled.set()

        def status(self):
            return SimpleNamespace(authenticated=False, expires_at=None)

        def logout(self) -> bool:
            return False

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "Prompt still works"

    async def drive(pilot: Pilot) -> None:
        status = pilot.app.query_one("#status", Static)
        await pilot.press(*"/login", "enter")
        async with asyncio.timeout(1):
            while not pilot.app.screen.query("#session-options"):
                await pilot.pause(0.01)
        await pilot.press("enter")
        async with asyncio.timeout(1):
            while not pilot.app.screen.query("#auth-code"):
                await pilot.pause(0.01)
        await pilot.press("ctrl+c")
        await asyncio.wait_for(login_cancelled.wait(), timeout=1)
        async with asyncio.timeout(1):
            while "idle" not in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press(*"After cancel", "enter")
        async with asyncio.timeout(1):
            while "↑0 ↓0" in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press(*"/quit", "enter")

    def terminal_factory(**kwargs) -> TextualTerminal:
        return TextualTerminal(**kwargs, auto_pilot=drive)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)
    monkeypatch.setattr(tui_app, "_AUTH_PROVIDER_LOADER", lambda: (Provider(),))
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
    response = (await store.load_session("test", session_id)).messages[-1]
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[-1], TextPart)
    assert response.parts[-1].content == "Prompt still works"


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
