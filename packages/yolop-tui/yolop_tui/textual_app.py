import asyncio
from collections.abc import Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea


class _PromptEditor(TextArea):
    BINDINGS = [Binding("enter", "submit", priority=True)]

    def __init__(self, *, on_submit: Callable[[str], None]) -> None:
        super().__init__(id="editor", show_line_numbers=False, soft_wrap=True)
        self._on_submit = on_submit

    def action_submit(self) -> None:
        text = self.text
        if not text.strip():
            return
        self.clear()
        self._on_submit(text)


class _YolopTextualApp(App[None]):
    ENABLE_COMMAND_PALETTE = False
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
        mounted: asyncio.Event,
    ) -> None:
        super().__init__()
        self._on_submit = on_submit
        self._terminal_ready = mounted

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="transcript"):
            yield Static(id="transcript-content")
        yield _PromptEditor(on_submit=self._on_submit)
        yield Static(id="status")

    def on_mount(self) -> None:
        self.query_one("#editor", TextArea).focus()
        self._terminal_ready.set()


class TextualTerminal:
    """Textual application facade used by the YoloP terminal host."""

    def __init__(self, *, on_submit: Callable[[str], None]) -> None:
        self._ready = asyncio.Event()
        self.app = _YolopTextualApp(on_submit=on_submit, mounted=self._ready)

    async def run(self) -> None:
        await self.app.run_async()

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    def stop(self) -> None:
        self.app.exit()
