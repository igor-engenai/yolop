from io import StringIO

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from rich.console import Console
from yolop_tui.rendering import Transcript


def _rich_plain(transcript: Transcript, width: int = 80) -> str:
    stream = StringIO()
    Console(file=stream, width=width, color_system=None).print(transcript.renderable())
    return stream.getvalue()


def test_transcript_provides_native_rich_markdown() -> None:
    transcript = Transcript()
    transcript.add_user("Question")
    transcript.add_assistant("# Answer\n\nUse **Rich** directly with `code`.")

    plain = _rich_plain(transcript, width=32)

    assert "› Question" in plain
    assert "Answer" in plain
    assert "Use Rich directly with code." in plain
    assert all(len(line) <= 32 for line in plain.splitlines())


def test_tool_output_is_compact_until_details_are_enabled() -> None:
    long_result = "x" * 9000
    transcript = Transcript.from_messages(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_file",
                        args={"path": "large.txt"},
                        tool_call_id="read",
                    )
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="read_file",
                        content=long_result,
                        tool_call_id="read",
                    )
                ]
            ),
            ModelResponse(parts=[TextPart("Done")]),
        ]
    )

    collapsed = _rich_plain(transcript)
    transcript.toggle_tools()
    expanded = _rich_plain(transcript)

    assert "read_file" in collapsed
    assert "success" in collapsed
    assert long_result[:50] not in collapsed
    assert long_result[:50] in expanded
    assert "[tool output truncated for display]" in expanded
    assert len(expanded) < len(long_result)


def test_native_renderable_rebuilds_completed_tools_and_thinking() -> None:
    transcript = Transcript.from_messages(
        [
            ModelResponse(
                parts=[
                    ThinkingPart("Private reasoning"),
                    ToolCallPart(
                        tool_name="read_file",
                        args={"path": "notes.md"},
                        tool_call_id="read",
                    ),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="read_file",
                        content="Completed detail",
                        tool_call_id="read",
                    )
                ]
            ),
        ]
    )

    collapsed = _rich_plain(transcript)
    transcript.toggle_tools()
    transcript.toggle_thinking()
    expanded = _rich_plain(transcript)

    assert "read_file" in collapsed
    assert "Completed detail" not in collapsed
    assert "Private reasoning" not in collapsed
    assert "Completed detail" in expanded
    assert "Private reasoning" in expanded


def test_transcript_wraps_in_a_narrow_terminal() -> None:
    transcript = Transcript()
    transcript.add_user("A long user prompt that must wrap")
    transcript.add_assistant("A long assistant response that must wrap")

    plain = _rich_plain(transcript, width=12)

    assert all(len(line) <= 12 for line in plain.splitlines())
