from rich.text import Text
from textual import events
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
