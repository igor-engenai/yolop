from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea
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
