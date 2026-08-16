from io import StringIO

from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from rich.console import Console
from wcwidth import wcswidth
from yolop_tui.rendering import Transcript


def _plain(transcript: Transcript, width: int = 80) -> str:
    return fragment_list_to_text(to_formatted_text(transcript.render(width)))


def _rich_plain(transcript: Transcript, width: int = 80) -> str:
    stream = StringIO()
    Console(file=stream, width=width, color_system=None).print(transcript.renderable())
    return stream.getvalue()


def test_transcript_provides_a_native_rich_renderable() -> None:
    transcript = Transcript()
    transcript.add_user("Question")
    transcript.add_assistant("# Answer\n\nUse **Rich** directly.")
    stream = StringIO()
    console = Console(file=stream, width=40, color_system=None)

    console.print(transcript.renderable())

    plain = stream.getvalue()
    assert "› Question" in plain
    assert "Answer" in plain
    assert "Use Rich directly." in plain


def test_markdown_assistant_output_is_styled_and_bounded_by_terminal_width() -> None:
    transcript = Transcript()
    transcript.add_assistant("# Heading\n\nThis has **bold text** and `code` in a long response.")

    fragments = to_formatted_text(transcript.render(32))
    plain = fragment_list_to_text(fragments)

    assert "Heading" in plain
    assert "bold text" in plain
    assert any(style for style, _text, *_mouse_handler in fragments)
    assert all(wcswidth(line) <= 32 for line in plain.splitlines())


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

    collapsed = _plain(transcript)
    transcript.toggle_tools()
    expanded = _plain(transcript)

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

    plain = _plain(transcript, width=12)

    assert all(wcswidth(line) <= 12 for line in plain.splitlines())
