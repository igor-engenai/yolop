import asyncio
from collections.abc import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension


class InlineTerminal:
    """Own the mutable terminal region while preserving normal scrollback."""

    def __init__(
        self,
        *,
        on_submit: Callable[[str], None],
        on_cancel: Callable[[], None] | None = None,
        completer: Completer | None = None,
    ) -> None:
        self._on_submit = on_submit
        self._on_cancel = on_cancel or (lambda: None)
        self._transcript = ""
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

    def set_transcript(self, text: str) -> None:
        self._transcript = text
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
        application = Application(
            layout=Layout(
                HSplit([transcript, top_border, editor_frame, bottom_border]),
                editor,
            ),
            key_bindings=bindings,
            full_screen=False,
            erase_when_done=False,
        )
        application.timeoutlen = 0.1
        application.ttimeoutlen = 0.1
        return application

    def _transcript_height(self) -> Dimension:
        if not self._transcript:
            return Dimension.exact(0)
        return Dimension(preferred=self._transcript.count("\n") + 1)


class _BorderControl(UIControl):
    def __init__(self, *, top: bool) -> None:
        self._top = top

    def create_content(self, width: int, height: int) -> UIContent:
        if self._top:
            label = "╭─ prompt "
            line = label + "─" * max(0, width - len(label) - 1) + "╮"
        else:
            line = "╰" + "─" * max(0, width - 2) + "╯"
        return UIContent(get_line=lambda _index: [("", line[:width])], line_count=1)
