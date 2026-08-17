import asyncio
from types import SimpleNamespace

from rich.text import Text
from textual import events
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.widgets import Input, OptionList, Static, TextArea
from yolop_tui.selection import SelectionOption
from yolop_tui.textual_app import TextualTerminal


async def test_textual_terminal_mounts_minimal_layout_and_submits_prompt() -> None:
    submitted: list[str] = []
    terminal = TextualTerminal(on_submit=submitted.append)

    async with terminal.app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        transcript = terminal.app.query_one("#transcript", VerticalScroll)
        editor = terminal.app.query_one("#editor", TextArea)
        status = terminal.app.query_one("#status", Static)

        assert transcript is not None
        assert editor.has_focus
        assert status is not None

        await pilot.press("h", "e", "l", "l", "o", "enter")
        await pilot.pause()

        assert submitted == ["hello"]
        assert editor.text == ""


async def test_textual_editor_supports_multiline_input_and_prompt_history() -> None:
    submitted: list[str] = []
    terminal = TextualTerminal(on_submit=submitted.append)

    async with terminal.app.run_test(size=(80, 24)) as pilot:
        editor = terminal.app.query_one("#editor", TextArea)

        await pilot.press("f", "i", "r", "s", "t", "enter")
        await pilot.press("l", "i", "n", "e", "1", "shift+enter", "l", "i", "n", "e", "2", "enter")
        await pilot.press("up")
        await pilot.pause()

        assert submitted == ["first", "line1\nline2"]
        assert editor.text == "line1\nline2"

        editor.clear()
        await pilot.press("a", "ctrl+j", "b", "enter")
        await pilot.pause()
        assert submitted[-1] == "a\nb"

        await pilot.press("t", "o", "p", "ctrl+j", "b", "o", "t", "t", "o", "m")
        await pilot.press("up", "X", "enter")
        await pilot.pause()
        assert submitted[-1] == "topX\nbottom"


async def test_textual_editor_interrupt_and_toggle_keys_are_context_sensitive() -> None:
    cancelled = 0
    toggled_tools = 0
    toggled_thinking = 0

    def cancel() -> bool:
        nonlocal cancelled
        cancelled += 1
        return True

    def toggle_tools() -> None:
        nonlocal toggled_tools
        toggled_tools += 1

    def toggle_thinking() -> None:
        nonlocal toggled_thinking
        toggled_thinking += 1

    terminal = TextualTerminal(
        on_submit=lambda _text: None,
        on_cancel=cancel,
        on_toggle_tools=toggle_tools,
        on_toggle_thinking=toggle_thinking,
    )

    async with terminal.app.run_test(size=(80, 24)) as pilot:
        editor = terminal.app.query_one("#editor", TextArea)
        await pilot.press("d", "i", "s", "c", "a", "r", "d", "ctrl+c")
        await pilot.press("ctrl+c", "escape", "ctrl+o", "ctrl+t")
        await pilot.pause()

        assert editor.text == ""
        assert cancelled == 2
        assert toggled_tools == 1
        assert toggled_thinking == 1
        assert terminal.app.is_running


async def test_textual_editor_completes_commands_and_project_files(tmp_path) -> None:
    source = tmp_path / "src" / "answer.py"
    source.parent.mkdir()
    source.write_text("ANSWER = 42\n")
    terminal = TextualTerminal(on_submit=lambda _text: None, cwd=tmp_path)

    async with terminal.app.run_test(size=(80, 24)) as pilot:
        editor = terminal.app.query_one("#editor", TextArea)
        completions = terminal.app.query_one("#completions", OptionList)

        await pilot.press("/", "r")
        await pilot.pause()
        assert completions.option_count == 1
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == "/resume"

        editor.clear()
        await pilot.press("@", "s", "r", "a")
        await pilot.pause()
        assert completions.option_count == 1
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == "@src/answer.py"


async def test_textual_session_picker_opens_from_external_run_task() -> None:
    picker_requested = asyncio.Event()

    async def drive(pilot: Pilot) -> None:
        await picker_requested.wait()
        await pilot.pause()
        await pilot.press("t", "g", "t", "enter")

    terminal = TextualTerminal(on_submit=lambda _text: None, auto_pilot=drive)
    options = [
        SelectionOption("first", "first-id  Earlier conversation"),
        SelectionOption("target", "target-id  Target conversation"),
    ]
    running = asyncio.create_task(terminal.run())

    try:
        await asyncio.wait_for(terminal.wait_until_ready(), timeout=1)
        selected = asyncio.create_task(terminal.choose(options))
        picker_requested.set()
        assert await asyncio.wait_for(selected, timeout=1) == "target"
    finally:
        terminal.stop()
        await running


async def test_textual_device_login_modal_shows_code_and_ctrl_c_cancels() -> None:
    cancelled = asyncio.Event()

    class Provider:
        name = "openai-codex"
        label = "OpenAI Codex (ChatGPT Plus/Pro)"

        async def login(self, notify):
            notify(
                SimpleNamespace(
                    verification_uri="https://auth.openai.com/codex/device",
                    user_code="ABCD-EFGH",
                    expires_in=900.0,
                )
            )
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        def status(self):
            return SimpleNamespace(authenticated=False, expires_at=None)

        def logout(self) -> bool:
            return False

    terminal = TextualTerminal(on_submit=lambda _text: None)

    async with terminal.app.run_test(size=(80, 24)) as pilot:
        login = asyncio.create_task(terminal.login_auth_provider(Provider()))
        await pilot.pause()

        assert "https://auth.openai.com/codex/device" in str(
            terminal.app.screen.query_one("#auth-uri", Static).render()
        )
        assert "ABCD-EFGH" in str(terminal.app.screen.query_one("#auth-code", Static).render())
        assert "Waiting" in str(terminal.app.screen.query_one("#auth-progress", Static).render())

        await pilot.press("ctrl+c")
        assert await asyncio.wait_for(login, timeout=1) is None
        await asyncio.wait_for(cancelled.wait(), timeout=1)


async def test_textual_confirmation_modal_accepts_and_cancels() -> None:
    terminal = TextualTerminal(on_submit=lambda _text: None)

    async with terminal.app.run_test(size=(80, 24)) as pilot:
        accepted = asyncio.create_task(terminal.confirm("Log out of openai-codex?"))
        await pilot.pause()
        assert "Log out of openai-codex?" in str(
            terminal.app.screen.query_one("#confirm-message", Static).render()
        )
        await pilot.press("y")
        assert await asyncio.wait_for(accepted, timeout=1) is True

        cancelled = asyncio.create_task(terminal.confirm("Log out?"))
        await pilot.pause()
        await pilot.press("escape")
        assert await asyncio.wait_for(cancelled, timeout=1) is False


async def test_textual_auth_provider_picker_uses_provider_labels() -> None:
    class Provider:
        name = "openai-codex"
        label = "OpenAI Codex (ChatGPT Plus/Pro)"

        async def login(self, notify):
            return SimpleNamespace(authenticated=True, expires_at=None)

        def status(self):
            return SimpleNamespace(authenticated=True, expires_at=None)

        def logout(self) -> bool:
            return False

    terminal = TextualTerminal(on_submit=lambda _text: None)

    async with terminal.app.run_test(size=(80, 24)) as pilot:
        selected = asyncio.create_task(terminal.choose_auth_provider([Provider()]))
        await pilot.pause()

        title = terminal.app.screen.query_one("#picker-title", Static)
        filter_input = terminal.app.screen.query_one("#session-filter", Input)
        options = terminal.app.screen.query_one("#session-options", OptionList)
        assert "Log in" in str(title.render())
        assert filter_input.placeholder == "Filter providers"
        assert options.get_option_at_index(0).prompt == "OpenAI Codex (ChatGPT Plus/Pro)"

        await pilot.press("enter")
        assert await asyncio.wait_for(selected, timeout=1) == "openai-codex"


async def test_textual_session_picker_fuzzy_selects_and_cancels() -> None:
    terminal = TextualTerminal(on_submit=lambda _text: None)
    options = [
        SelectionOption("first", "first-id  Earlier conversation"),
        SelectionOption("target", "target-id  Target conversation"),
    ]

    async with terminal.app.run_test(size=(80, 24)) as pilot:
        selected = asyncio.create_task(terminal.choose(options))
        await pilot.pause()
        filter_input = terminal.app.screen.query_one("#session-filter", Input)
        option_list = terminal.app.screen.query_one("#session-options", OptionList)
        await pilot.press("t", "g", "t")
        await pilot.pause()
        assert filter_input.value == "tgt"
        assert option_list.option_count == 1
        await pilot.press("enter")
        assert await asyncio.wait_for(selected, timeout=1) == "target"

        cancelled = asyncio.create_task(terminal.choose(options))
        await pilot.pause()
        await pilot.press("escape")
        assert await asyncio.wait_for(cancelled, timeout=1) is None

        clicked = asyncio.create_task(terminal.choose(options))
        await pilot.pause()
        assert await pilot.click("#session-options", offset=(2, 1))
        assert await asyncio.wait_for(clicked, timeout=1) == "first"


async def test_textual_ctrl_c_and_ctrl_d_exit_when_idle_and_empty() -> None:
    for key in ("ctrl+c", "ctrl+d"):
        terminal = TextualTerminal(on_submit=lambda _text: None, on_cancel=lambda: False)

        async with terminal.app.run_test(size=(80, 24)) as pilot:
            await pilot.press(key)
            await pilot.pause()
            assert not terminal.app.is_running


async def test_textual_transcript_scrolls_and_follows_only_at_bottom() -> None:
    terminal = TextualTerminal(on_submit=lambda _text: None)
    lines = [f"history line {index}" for index in range(40)]
    terminal.set_transcript(Text("\n".join(lines)))
    terminal.set_status("idle")

    async with terminal.app.run_test(size=(80, 16)) as pilot:
        await pilot.pause()
        scroll = terminal.app.query_one("#transcript", VerticalScroll)
        status = terminal.app.query_one("#status", Static)

        assert scroll.scroll_y == scroll.max_scroll_y
        assert "idle" in str(status.render())

        scroll.post_message(
            events.MouseScrollUp(
                scroll,
                1,
                1,
                0,
                -1,
                0,
                False,
                False,
                False,
            )
        )
        await pilot.pause()
        assert scroll.scroll_y < scroll.max_scroll_y
        mouse_position = scroll.scroll_y
        await pilot.resize_terminal(80, 15)
        await pilot.pause(0.05)
        assert scroll.scroll_y == mouse_position
        await pilot.press("end")
        await pilot.pause()

        await pilot.press("pageup")
        await pilot.pause()
        paused_position = scroll.scroll_y
        assert paused_position < scroll.max_scroll_y

        terminal.set_transcript(Text("\n".join([*lines, "new while paused"])))
        await pilot.pause()
        assert scroll.scroll_y == paused_position

        await pilot.press("end")
        await pilot.pause()
        assert scroll.scroll_y == scroll.max_scroll_y

        terminal.set_transcript(Text("\n".join([*lines, "one", "two"])))
        await pilot.pause(0.05)
        assert scroll.scroll_y == scroll.max_scroll_y

        await pilot.resize_terminal(60, 10)
        await pilot.pause(0.05)
        assert scroll.scroll_y == scroll.max_scroll_y
