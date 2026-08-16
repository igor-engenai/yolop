import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import AnyFormattedText, fragment_list_to_text, to_formatted_text
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.styles import Style

from .selection import SelectionOption


@dataclass
class _Selection:
    options: tuple[SelectionOption, ...]
    future: asyncio.Future[str | None]
    selected: int = 0


class InlineTerminal:
    """Own the mutable terminal region while preserving normal scrollback."""

    def __init__(
        self,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], bool | None] | None = None,
        on_toggle_tools: Callable[[], None] | None = None,
        on_toggle_thinking: Callable[[], None] | None = None,
        completer: Completer | None = None,
    ) -> None:
        self._on_submit = on_submit
        self._on_cancel = on_cancel or (lambda: None)
        self._on_toggle_tools = on_toggle_tools or (lambda: None)
        self._on_toggle_thinking = on_toggle_thinking or (lambda: None)
        self._transcript: AnyFormattedText = ""
        self._transcript_lines = 0
        self._transcript_scroll = 0
        self._status = ""
        self._selection: _Selection | None = None
        self._ready = asyncio.Event()
        self._buffer = Buffer(
            completer=completer,
            complete_while_typing=completer is not None,
            enable_history_search=True,
            history=InMemoryHistory(),
            multiline=True,
            on_text_changed=lambda _buffer: self._selection_query_changed(),
        )
        self._application = self._build_application()

    async def run(self) -> None:
        self._ready.set()
        try:
            await self._application.run_async()
        finally:
            self._application.output.disable_mouse_support()
            self._application.output.flush()

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    async def choose(self, options: list[SelectionOption]) -> str | None:
        if self._selection is not None:
            raise RuntimeError("A terminal selection is already active")
        future = asyncio.get_running_loop().create_future()
        selection = _Selection(tuple(options), future)
        self._selection = selection
        self._buffer.reset()
        self._application.invalidate()
        try:
            return await future
        finally:
            if self._selection is selection:
                self._selection = None
                self._buffer.reset()
                self._application.invalidate()

    @property
    def width(self) -> int:
        return self._application.output.get_size().columns

    def set_transcript(self, text: AnyFormattedText) -> None:
        previous_lines = self._transcript_lines
        self._transcript = text
        plain = fragment_list_to_text(to_formatted_text(text))
        self._transcript_lines = plain.count("\n") + bool(plain)
        if self._transcript_scroll:
            self._transcript_scroll += max(0, self._transcript_lines - previous_lines)
            self._transcript_scroll = min(
                self._transcript_scroll,
                self._max_transcript_scroll(),
            )
        self._application.invalidate()

    def set_status(self, text: str) -> None:
        self._status = text
        self._application.invalidate()

    def restore_editor_text(self, text: str) -> None:
        if not text:
            return
        current = self._buffer.text
        restored = f"{current}\n\n{text}" if current else text
        self._buffer.set_document(Document(restored, cursor_position=len(restored)))
        self._application.invalidate()

    def stop(self) -> None:
        if self._application.is_running:
            self._application.exit()

    def _build_application(self) -> Application[None]:
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(_event) -> None:
            if self._selection is not None:
                self._finish_selection()
                return
            text = self._buffer.text
            if not text.strip():
                return
            if not text.lstrip().startswith("/"):
                self._buffer.append_to_history()
            self._buffer.reset()
            self._on_submit(text)

        @bindings.add("c-j")
        def newline(_event) -> None:
            if self._selection is None:
                self._buffer.insert_text("\n")

        @bindings.add("pageup")
        def transcript_page_up(_event) -> None:
            self._scroll_transcript(self._transcript_page_size())

        @bindings.add("pagedown")
        def transcript_page_down(_event) -> None:
            self._scroll_transcript(-self._transcript_page_size())

        @bindings.add("end")
        def transcript_end(_event) -> None:
            self._scroll_transcript_to_bottom()

        @bindings.add("up")
        def up(_event) -> None:
            if self._selection is not None:
                self._move_selection(-1)
            elif self._buffer.document.cursor_position_row == 0:
                self._buffer.history_backward()
            else:
                self._buffer.cursor_up()

        @bindings.add("down")
        def down(_event) -> None:
            if self._selection is not None:
                self._move_selection(1)
            elif self._buffer.document.cursor_position_row == self._buffer.document.line_count - 1:
                self._buffer.history_forward()
            else:
                self._buffer.cursor_down()

        @bindings.add("escape")
        def cancel(_event) -> None:
            if self._selection is not None:
                self._cancel_selection()
            else:
                self._on_cancel()

        @bindings.add("c-c")
        def interrupt(_event) -> None:
            if self._buffer.text:
                self._buffer.reset()
            elif self._on_cancel() is not True:
                self.stop()

        @bindings.add("c-o")
        def toggle_tools(_event) -> None:
            self._on_toggle_tools()

        @bindings.add("c-t")
        def toggle_thinking(_event) -> None:
            self._on_toggle_thinking()

        @bindings.add("c-d")
        def exit_when_empty(_event) -> None:
            if not self._buffer.text and self._selection is None:
                self.stop()

        transcript = Window(
            content=FormattedTextControl(
                self._transcript_fragments,
                get_cursor_position=self._transcript_cursor_position,
            ),
            height=self._transcript_height,
            wrap_lines=True,
            get_vertical_scroll=self._transcript_vertical_scroll,
            always_hide_cursor=True,
        )
        self._transcript_window = transcript
        selector = Window(
            content=FormattedTextControl(self._selection_fragments),
            height=self._selection_height,
            wrap_lines=False,
        )
        top_border = Window(
            content=_BorderControl(top=True, label=self._editor_label),
            height=1,
        )
        editor = Window(
            content=BufferControl(buffer=self._buffer),
            height=Dimension(min=1, max=8, preferred=1),
            wrap_lines=True,
        )
        editor_frame = VSplit(
            [
                Window(content=FormattedTextControl("│"), width=1),
                editor,
                Window(content=FormattedTextControl("│"), width=1),
            ]
        )
        bottom_border = Window(content=_BorderControl(top=False), height=1)
        status = Window(
            content=FormattedTextControl(lambda: [("class:status", self._status)]),
            height=1,
            wrap_lines=False,
        )
        application = Application(
            layout=Layout(
                HSplit([transcript, selector, top_border, editor_frame, bottom_border, status]),
                editor,
            ),
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "border": "ansibrightblack",
                    "selection": "reverse",
                    "selection.muted": "ansibrightblack",
                    "status": "ansibrightblack",
                }
            ),
            full_screen=False,
            mouse_support=True,
            erase_when_done=False,
        )
        application.timeoutlen = 0.1
        application.ttimeoutlen = 0.1
        return application

    def _transcript_fragments(self) -> AnyFormattedText:
        return [
            (style, text, self._handle_transcript_mouse)
            for style, text, *_rest in to_formatted_text(self._transcript)
        ]

    def _handle_transcript_mouse(
        self,
        mouse_event: MouseEvent,
    ) -> object:
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self._scroll_transcript(1)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self._scroll_transcript(-1)
            return None
        return NotImplemented

    def _transcript_cursor_position(self) -> Point:
        if not self._transcript_scroll:
            return Point(x=0, y=max(0, self._transcript_lines - 1))
        return Point(x=0, y=self._transcript_vertical_scroll(self._transcript_window))

    def _transcript_vertical_scroll(self, _window: Window) -> int:
        return max(
            0,
            self._transcript_lines - self._transcript_page_size() - self._transcript_scroll,
        )

    def _transcript_page_size(self) -> int:
        render_info = self._transcript_window.render_info
        if render_info is not None:
            return max(1, render_info.window_height)
        return max(1, self._application.output.get_size().rows - 4)

    def _max_transcript_scroll(self) -> int:
        return max(0, self._transcript_lines - self._transcript_page_size())

    def _scroll_transcript(self, lines: int) -> None:
        self._transcript_scroll = min(
            self._max_transcript_scroll(),
            max(0, self._transcript_scroll + lines),
        )
        self._application.invalidate()

    def _scroll_transcript_to_bottom(self) -> None:
        self._transcript_scroll = 0
        self._application.invalidate()

    def _transcript_height(self) -> Dimension:
        return Dimension(
            min=0,
            preferred=self._transcript_lines,
            max=self._transcript_lines,
        )

    def _selection_height(self) -> Dimension:
        if self._selection is None:
            return Dimension.exact(0)
        return Dimension.exact(max(1, min(5, len(self._selection_matches()))))

    def _selection_fragments(self) -> list[tuple[str, str]]:
        selection = self._selection
        if selection is None:
            return []
        matches = self._selection_matches()
        if not matches:
            return [("class:selection.muted", "  No matching sessions")]
        selection.selected = min(selection.selected, len(matches) - 1)
        start = max(0, min(selection.selected, len(matches) - 5))
        fragments: list[tuple[str, str]] = []
        for index, option in enumerate(matches[start : start + 5], start=start):
            style = "class:selection" if index == selection.selected else ""
            prefix = "> " if index == selection.selected else "  "
            fragments.append((style, prefix + option.label))
            if index < min(start + 4, len(matches) - 1):
                fragments.append(("", "\n"))
        return fragments

    def _selection_matches(self) -> list[SelectionOption]:
        if self._selection is None:
            return []
        query = self._buffer.text.casefold()
        return [
            option
            for option in self._selection.options
            if _fuzzy_match(query, option.label.casefold())
        ]

    def _selection_query_changed(self) -> None:
        if self._selection is not None:
            self._selection.selected = 0
            self._application.invalidate()

    def _move_selection(self, direction: int) -> None:
        assert self._selection is not None
        matches = self._selection_matches()
        if matches:
            self._selection.selected = max(
                0,
                min(len(matches) - 1, self._selection.selected + direction),
            )
            self._application.invalidate()

    def _finish_selection(self) -> None:
        assert self._selection is not None
        matches = self._selection_matches()
        if not matches:
            return
        value = matches[self._selection.selected].value
        selection = self._selection
        self._selection = None
        self._buffer.reset()
        if not selection.future.done():
            selection.future.set_result(value)
        self._application.invalidate()

    def _cancel_selection(self) -> None:
        assert self._selection is not None
        selection = self._selection
        self._selection = None
        self._buffer.reset()
        if not selection.future.done():
            selection.future.set_result(None)
        self._application.invalidate()

    def _editor_label(self) -> str:
        return "resume" if self._selection is not None else "prompt"


class _BorderControl(UIControl):
    def __init__(self, *, top: bool, label: Callable[[], str] | None = None) -> None:
        self._top = top
        self._label = label or (lambda: "")

    def create_content(self, width: int, height: int) -> UIContent:
        if self._top:
            label = f"╭─ {self._label()} "
            line = label + "─" * max(0, width - len(label) - 1) + "╮"
        else:
            line = "╰" + "─" * max(0, width - 2) + "╯"
        return UIContent(get_line=lambda _index: [("class:border", line[:width])], line_count=1)


def _fuzzy_match(query: str, value: str) -> bool:
    characters = iter(value)
    return all(any(candidate == expected for candidate in characters) for expected in query)
