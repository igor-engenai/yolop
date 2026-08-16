import asyncio

from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from yolop_tui.terminal import InlineTerminal


class CapturingOutput(DummyOutput):
    def __init__(self, *, columns: int = 80, rows: int = 24) -> None:
        self.columns = columns
        self.rows = rows
        self.writes: list[str] = []

    def write(self, data: str) -> None:
        self.writes.append(data)

    def write_raw(self, data: str) -> None:
        self.writes.append(data)

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=self.columns)

    @property
    def text(self) -> str:
        return "".join(self.writes)


async def test_inline_terminal_keeps_multiline_input_during_async_output() -> None:
    submitted: list[str] = []
    output = CapturingOutput()

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            terminal = InlineTerminal(on_submit=submitted.append)
            running = asyncio.create_task(terminal.run())
            await terminal.wait_until_ready()

            pipe_input.send_text("first")
            await asyncio.sleep(0)
            terminal.set_transcript("assistant output")
            pipe_input.send_bytes(b"\n")
            pipe_input.send_text("second\r")

            async with asyncio.timeout(1):
                while not submitted:
                    await asyncio.sleep(0)
            terminal.stop()
            await running

    assert submitted == ["first\nsecond"]
    assert "assistant output" in output.text
    assert "╭─ prompt" in output.text
    assert "\x1b[?1049h" not in output.text


async def test_inline_terminal_redraws_mutable_content_after_resize() -> None:
    output = CapturingOutput(columns=64)

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            terminal = InlineTerminal(on_submit=lambda _text: None)
            running = asyncio.create_task(terminal.run())
            await terminal.wait_until_ready()

            terminal.set_transcript("tool result: collapsed")
            await asyncio.sleep(0.01)
            output.columns = 32
            terminal.set_transcript("tool result: expanded\nfull output")
            await asyncio.sleep(0.01)
            terminal.stop()
            await running

    assert "tool result: collapsed" in output.text
    assert "tool result: expanded" in output.text
    assert "full output" in output.text
    normalized = output.text.replace("\r", "")
    assert "╭─ prompt " + "─" * 21 + "╮" in normalized


async def test_ctrl_c_clears_and_ctrl_d_exits_only_with_empty_editor() -> None:
    submitted: list[str] = []
    output = CapturingOutput()

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            terminal = InlineTerminal(on_submit=submitted.append)
            running = asyncio.create_task(terminal.run())
            await terminal.wait_until_ready()

            pipe_input.send_text("discard")
            pipe_input.send_bytes(b"\x03")
            pipe_input.send_text("\r")
            await asyncio.sleep(0.01)
            assert submitted == []

            pipe_input.send_text("keep")
            pipe_input.send_bytes(b"\x04")
            await asyncio.sleep(0.01)
            assert not running.done()
            pipe_input.send_bytes(b"\x03\x04")
            await asyncio.wait_for(running, timeout=1)

    assert submitted == []


async def test_up_moves_within_multiline_input_then_reads_prompt_history() -> None:
    submitted: list[str] = []
    output = CapturingOutput()

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            terminal = InlineTerminal(on_submit=submitted.append)
            running = asyncio.create_task(terminal.run())
            await terminal.wait_until_ready()

            pipe_input.send_text("first prompt\r")
            await asyncio.sleep(0.01)
            pipe_input.send_text("line1\nline2")
            pipe_input.send_bytes(b"\x1b[A")
            pipe_input.send_text("X\r")
            await asyncio.sleep(0.01)
            pipe_input.send_bytes(b"\x1b[A")
            pipe_input.send_text("\r")
            await asyncio.sleep(0.01)
            terminal.stop()
            await running

    assert submitted == ["first prompt", "line1X\nline2", "line1X\nline2"]
