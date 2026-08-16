import asyncio
from collections.abc import Callable

from rich.console import RenderableType
from rich.text import Text
from textual import events
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
        self._transcript_renderable: RenderableType = Text()
        self._status_text = ""
        self._follow_transcript = True

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="transcript"):
            yield Static(self._transcript_renderable, id="transcript-content")
        yield _PromptEditor(on_submit=self._on_submit)
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


class TextualTerminal:
    """Textual application facade used by the YoloP terminal host."""

    def __init__(self, *, on_submit: Callable[[str], None]) -> None:
        self._ready = asyncio.Event()
        self.app = _YolopTextualApp(on_submit=on_submit, mounted=self._ready)

    async def run(self) -> None:
        await self.app.run_async()

    async def wait_until_ready(self) -> None:
        await self._ready.wait()

    def set_transcript(self, renderable: RenderableType) -> None:
        self.app.set_transcript(renderable)

    def set_status(self, text: str) -> None:
        self.app.set_status(text)

    def stop(self) -> None:
        self.app.exit()
