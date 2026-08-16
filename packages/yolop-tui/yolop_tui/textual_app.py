from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from rich.console import RenderableType
from rich.text import Text
from textual import events
from textual.app import App, AutopilotCallbackType, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from .selection import SelectionOption
from .suggestions import PromptCompleter, PromptCompletion


class _PromptEditor(TextArea):
    BINDINGS = [
        Binding("enter", "submit", priority=True),
        Binding("shift+enter", "newline", priority=True),
        Binding("ctrl+j", "newline", priority=True),
        Binding("up", "history_up", priority=True),
        Binding("down", "history_down", priority=True),
        Binding("ctrl+c", "interrupt", priority=True),
        Binding("ctrl+d", "eof", priority=True),
        Binding("escape", "cancel", priority=True),
        Binding("ctrl+o", "toggle_tools", priority=True),
        Binding("ctrl+t", "toggle_thinking", priority=True),
        Binding("tab", "accept_completion", priority=True),
    ]

    def __init__(
        self,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], bool | None],
        on_toggle_tools: Callable[[], None],
        on_toggle_thinking: Callable[[], None],
        on_exit: Callable[[], None],
        on_move_completion: Callable[[int], bool],
        on_accept_completion: Callable[[], bool],
    ) -> None:
        super().__init__(id="editor", show_line_numbers=False, soft_wrap=True)
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._on_toggle_tools = on_toggle_tools
        self._on_toggle_thinking = on_toggle_thinking
        self._on_exit = on_exit
        self._on_move_completion = on_move_completion
        self._on_accept_completion = on_accept_completion
        self._history: list[str] = []
        self._history_position: int | None = None
        self._history_draft = ""

    def action_submit(self) -> None:
        text = self.text
        if not text.strip():
            return
        if not text.lstrip().startswith("/"):
            self._history.append(text)
        self._history_position = None
        self._history_draft = ""
        self.clear()
        self._on_submit(text)

    def action_newline(self) -> None:
        self.insert("\n")

    def action_history_up(self) -> None:
        if self._on_move_completion(-1):
            return
        row, _column = self.cursor_location
        if row > 0:
            self.action_cursor_up()
            return
        if not self._history:
            return
        if self._history_position is None:
            self._history_draft = self.text
            self._history_position = len(self._history) - 1
        elif self._history_position > 0:
            self._history_position -= 1
        self._load_history_text(self._history[self._history_position])

    def action_history_down(self) -> None:
        if self._on_move_completion(1):
            return
        row, _column = self.cursor_location
        if row < self.text.count("\n"):
            self.action_cursor_down()
            return
        if self._history_position is None:
            return
        if self._history_position < len(self._history) - 1:
            self._history_position += 1
            text = self._history[self._history_position]
        else:
            self._history_position = None
            text = self._history_draft
        self._load_history_text(text)

    def action_interrupt(self) -> None:
        if self.text:
            self.clear()
        elif self._on_cancel() is not True:
            self._on_exit()

    def action_eof(self) -> None:
        if not self.text:
            self._on_exit()
        else:
            self.action_delete_right()

    def action_cancel(self) -> None:
        self._on_cancel()

    def action_toggle_tools(self) -> None:
        self._on_toggle_tools()

    def action_toggle_thinking(self) -> None:
        self._on_toggle_thinking()

    def action_accept_completion(self) -> None:
        self._on_accept_completion()

    def restore_text(self, text: str) -> None:
        if not text:
            return
        current = self.text
        self._load_history_text(f"{current}\n\n{text}" if current else text)

    def _load_history_text(self, text: str) -> None:
        self.load_text(text)
        lines = text.split("\n")
        self.move_cursor((len(lines) - 1, len(lines[-1])))


class _SessionPicker(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", priority=True),
        Binding("enter", "select", priority=True),
        Binding("up", "cursor_up", priority=True),
        Binding("down", "cursor_down", priority=True),
    ]
    CSS = """
    _SessionPicker {
        align: center middle;
        background: $background 70%;
    }

    #session-picker {
        width: 80%;
        height: 60%;
        min-height: 8;
        border: round ansi_bright_black;
        background: $surface;
        padding: 1;
    }

    #session-filter {
        width: 1fr;
        height: 3;
        border: round ansi_bright_black;
    }

    #session-options {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, options: list[SelectionOption]) -> None:
        super().__init__()
        self._options = tuple(options)
        self._matches = self._options

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker"):
            yield Input(placeholder="Filter sessions", id="session-filter")
            yield OptionList(id="session-options")

    def on_mount(self) -> None:
        self.query_one("#session-filter", Input).focus()
        self._refresh_options("")

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh_options(event.value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        options = self.query_one("#session-options", OptionList)
        if options.option_count == 0:
            return
        index = options.highlighted or 0
        option = options.get_option_at_index(index)
        if option.id is not None:
            self.dismiss(option.id)

    def action_cursor_up(self) -> None:
        self.query_one("#session-options", OptionList).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#session-options", OptionList).action_cursor_down()

    def _refresh_options(self, query: str) -> None:
        self._matches = tuple(
            option
            for option in self._options
            if _fuzzy_match(query.casefold(), option.label.casefold())
        )
        options = self.query_one("#session-options", OptionList)
        options.clear_options()
        options.add_options(Option(option.label, id=option.value) for option in self._matches)
        options.highlighted = 0 if self._matches else None


class _YolopTextualApp(App[None]):
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("pageup", "transcript_page_up", priority=True),
        Binding("pagedown", "transcript_page_down", priority=True),
        Binding("end", "transcript_end", priority=True),
    ]
    CSS = """
    Screen {
        layout: vertical;
    }

    #transcript {
        height: 1fr;
        overflow-y: auto;
        scrollbar-size: 1 1;
    }

    #transcript-content {
        width: 1fr;
        height: auto;
    }

    #completions {
        display: none;
        width: 1fr;
        height: auto;
        max-height: 6;
        border: solid ansi_bright_black;
    }

    #editor {
        width: 1fr;
        height: auto;
        min-height: 3;
        max-height: 10;
        border: round ansi_bright_black;
        padding: 0 1;
    }

    #status {
        width: 1fr;
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], bool | None],
        on_toggle_tools: Callable[[], None],
        on_toggle_thinking: Callable[[], None],
        mounted: asyncio.Event,
        cwd: Path,
    ) -> None:
        super().__init__()
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._on_toggle_tools = on_toggle_tools
        self._on_toggle_thinking = on_toggle_thinking
        self._terminal_ready = mounted
        self._completer = PromptCompleter(cwd)
        self._completions: tuple[PromptCompletion, ...] = ()
        self._transcript_renderable: RenderableType = Text()
        self._status_text = ""
        self._follow_transcript = True

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="transcript"):
            yield Static(self._transcript_renderable, id="transcript-content")
        yield OptionList(id="completions")
        yield _PromptEditor(
            on_submit=self._on_submit,
            on_cancel=self._on_cancel,
            on_toggle_tools=self._on_toggle_tools,
            on_toggle_thinking=self._on_toggle_thinking,
            on_exit=self.exit,
            on_move_completion=self._move_completion,
            on_accept_completion=self._accept_completion,
        )
        yield Static(self._status_text, id="status")

    def on_mount(self) -> None:
        self.query_one("#editor", TextArea).focus()
        self.call_after_refresh(self.action_transcript_end)
        self._terminal_ready.set()

    def on_resize(self, _event: events.Resize) -> None:
        if self._follow_transcript:
            self.run_worker(
                self._follow_after_layout(),
                group="transcript-follow",
                exclusive=True,
            )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        editor = event.text_area
        cursor = _cursor_offset(editor)
        self._completions = self._completer.complete(editor.text[:cursor])
        options = self.query_one("#completions", OptionList)
        options.clear_options()
        if not self._completions:
            options.styles.display = "none"
            return
        options.add_options(
            Option(completion.value, id=str(index))
            for index, completion in enumerate(self._completions)
        )
        options.highlighted = 0
        options.styles.display = "block"

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "completions" and event.option.id is not None:
            self._accept_completion(int(event.option.id))

    def _move_completion(self, direction: int) -> bool:
        if not self._completions:
            return False
        options = self.query_one("#completions", OptionList)
        current = options.highlighted or 0
        options.highlighted = max(0, min(len(self._completions) - 1, current + direction))
        return True

    def _accept_completion(self, index: int | None = None) -> bool:
        if not self._completions:
            return False
        options = self.query_one("#completions", OptionList)
        selected = index if index is not None else (options.highlighted or 0)
        completion = self._completions[selected]
        editor = self.query_one("#editor", _PromptEditor)
        cursor = _cursor_offset(editor)
        text = editor.text
        editor._load_history_text(text[: completion.start] + completion.value + text[cursor:])
        self._completions = ()
        options.clear_options()
        options.styles.display = "none"
        return True

    def set_transcript(self, renderable: RenderableType) -> None:
        self._transcript_renderable = renderable
        if not self.is_running:
            return
        transcript = self.query_one("#transcript", VerticalScroll)
        self._follow_transcript = transcript.is_vertical_scroll_end
        content = self.query_one("#transcript-content", Static)
        content.update(renderable)
        if self._follow_transcript:
            self.run_worker(
                self._follow_after_refresh(content),
                group="transcript-follow",
                exclusive=True,
            )

    def set_status(self, text: str) -> None:
        self._status_text = text
        if self.is_running:
            self.query_one("#status", Static).update(text)

    async def _follow_after_refresh(self, content: Static) -> None:
        await content.wait_for_refresh()
        await self._follow_after_layout()

    async def _follow_after_layout(self) -> None:
        await asyncio.sleep(0.01)
        if self._follow_transcript and self.is_running:
            self.action_transcript_end()

    def action_transcript_page_up(self) -> None:
        self._follow_transcript = False
        self.query_one("#transcript", VerticalScroll).scroll_page_up(animate=False)

    def action_transcript_page_down(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_page_down(animate=False)

    def action_transcript_end(self) -> None:
        self._follow_transcript = True
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)


def _cursor_offset(editor: TextArea) -> int:
    row, column = editor.cursor_location
    lines = editor.text.split("\n")
    return sum(len(line) + 1 for line in lines[:row]) + column


class TextualTerminal:
    """Textual application facade used by the YoloP terminal host."""

    def __init__(
        self,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], bool | None] | None = None,
        on_toggle_tools: Callable[[], None] | None = None,
        on_toggle_thinking: Callable[[], None] | None = None,
        cwd: Path | None = None,
        auto_pilot: AutopilotCallbackType | None = None,
    ) -> None:
        self._ready = asyncio.Event()
        self._auto_pilot = auto_pilot
        self.app = _YolopTextualApp(
            on_submit=on_submit,
            on_cancel=on_cancel or (lambda: None),
            on_toggle_tools=on_toggle_tools or (lambda: None),
            on_toggle_thinking=on_toggle_thinking or (lambda: None),
            mounted=self._ready,
            cwd=(cwd or Path.cwd()).resolve(),
        )

    async def run(self) -> None:
        await self.app.run_async(
            headless=self._auto_pilot is not None,
            auto_pilot=self._auto_pilot,
        )

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    async def choose(self, options: list[SelectionOption]) -> str | None:
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()

        def selected(value: str | None) -> None:
            if not future.done():
                future.set_result(value)

        self.app.push_screen(_SessionPicker(options), callback=selected)
        return await future

    def set_transcript(self, renderable: RenderableType) -> None:
        self.app.set_transcript(renderable)

    def set_status(self, text: str) -> None:
        self.app.set_status(text)

    def restore_editor_text(self, text: str) -> None:
        if self.app.is_running:
            self.app.query_one("#editor", _PromptEditor).restore_text(text)

    def stop(self) -> None:
        self.app.exit()


def _fuzzy_match(query: str, value: str) -> bool:
    characters = iter(value)
    return all(any(candidate == expected for candidate in characters) for expected in query)
