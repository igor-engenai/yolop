import asyncio
from collections.abc import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import AnyFormattedText, fragment_list_to_text, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style


class InlineTerminal:
    """Own the mutable terminal region while preserving normal scrollback."""

    def __init__(
        self,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], None] | None = None,
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
        self._status = ""
        self._ready = asyncio.Event()
        self._buffer = Buffer(
            completer=completer,
            complete_while_typing=completer is not None,
            multiline=True,
        )
        self._application = self._build_application()

    async def run(self) -> None:
        self._ready.set()
        await self._application.run_async()

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    @property
    def width(self) -> int:
        return self._application.output.get_size().columns

    def set_transcript(self, text: AnyFormattedText) -> None:
        self._transcript = text
        plain = fragment_list_to_text(to_formatted_text(text))
        self._transcript_lines = plain.count("\n") + bool(plain)
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
            text = self._buffer.text
            if not text.strip():
                return
            self._buffer.reset()
            self._on_submit(text)

        @bindings.add("c-j")
        def newline(_event) -> None:
            self._buffer.insert_text("\n")

        @bindings.add("escape")
        def cancel(_event) -> None:
            self._on_cancel()

        @bindings.add("c-c")
        def clear_editor(_event) -> None:
            self._buffer.reset()

        @bindings.add("c-o")
        def toggle_tools(_event) -> None:
            self._on_toggle_tools()

        @bindings.add("c-t")
        def toggle_thinking(_event) -> None:
            self._on_toggle_thinking()

        @bindings.add("c-d")
        def exit_when_empty(_event) -> None:
            if not self._buffer.text:
                self.stop()

        transcript = Window(
            content=FormattedTextControl(lambda: self._transcript),
            height=self._transcript_height,
            wrap_lines=True,
        )
        top_border = Window(content=_BorderControl(top=True), height=1)
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
                HSplit([transcript, top_border, editor_frame, bottom_border, status]),
                editor,
            ),
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "border": "ansibrightblack",
                    "status": "ansibrightblack",
                }
            ),
            full_screen=False,
            erase_when_done=False,
        )
        application.timeoutlen = 0.1
        application.ttimeoutlen = 0.1
        return application

    def _transcript_height(self) -> Dimension:
        return Dimension.exact(self._transcript_lines)


class _BorderControl(UIControl):
    def __init__(self, *, top: bool) -> None:
        self._top = top

    def create_content(self, width: int, height: int) -> UIContent:
        if self._top:
            label = "╭─ prompt "
            line = label + "─" * max(0, width - len(label) - 1) + "╮"
        else:
            line = "╰" + "─" * max(0, width - 2) + "╯"
        return UIContent(get_line=lambda _index: [("class:border", line[:width])], line_count=1)
