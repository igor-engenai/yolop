import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai import AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextContent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text

_MAX_TOOL_DETAIL_CHARS = 8192


@dataclass
class _TextEntry:
    role: str
    text: str


@dataclass
class _ToolEntry:
    tool_call_id: str
    name: str
    arguments: str
    status: str = "running"
    result: str | None = None


class Transcript:
    """Project native messages and events into the mutable terminal transcript."""

    def __init__(self) -> None:
        self._entries: list[_TextEntry | _ToolEntry] = []
        self._tools: dict[str, _ToolEntry] = {}
        self.show_tools = False
        self.show_thinking = False

    @classmethod
    def from_messages(cls, messages: list[ModelMessage]) -> "Transcript":
        transcript = cls()
        transcript.reset(messages)
        return transcript

    def reset(self, messages: list[ModelMessage]) -> None:
        self._entries = []
        self._tools = {}
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        text = _display_user_content(part.content)
                        if text:
                            self._entries.append(_TextEntry("user", text))
                    elif isinstance(part, ToolReturnPart | RetryPromptPart):
                        self._add_tool_result(part)
            elif isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, TextPart):
                        self.add_assistant(part.content)
                    elif isinstance(part, ThinkingPart):
                        self._append_text("thinking", part.content)
                    elif isinstance(part, ToolCallPart):
                        self._add_tool_call(part)
        for tool in self._tools.values():
            if tool.status == "running":
                tool.status = "interrupted"

    def apply(self, event: AgentStreamEvent) -> bool:
        if isinstance(event, FunctionToolCallEvent):
            self._add_tool_call(event.part)
        elif isinstance(event, FunctionToolResultEvent):
            self._add_tool_result(event.part)
        elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            self.add_assistant(event.part.content)
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            self.add_assistant(event.delta.content_delta)
        elif isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            self._append_text("thinking", event.part.content)
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
            if event.delta.content_delta:
                self._append_text("thinking", event.delta.content_delta)
        else:
            return False
        return True

    def add_user(self, text: str) -> None:
        self._entries.append(_TextEntry("user", text))

    def add_assistant(self, text: str) -> None:
        self._append_text("assistant", text)

    def add_error(self, text: str) -> None:
        self._entries.append(_TextEntry("error", text))

    def add_notice(self, text: str) -> None:
        self._entries.append(_TextEntry("notice", text))

    def toggle_tools(self) -> None:
        self.show_tools = not self.show_tools

    def toggle_thinking(self) -> None:
        self.show_thinking = not self.show_thinking

    def renderable(self) -> Group:
        """Build a native Rich projection for managed terminal hosts."""
        renderables: list[RenderableType] = []
        for entry in self._visible_entries():
            if renderables:
                renderables.append(Text(""))
            if isinstance(entry, _ToolEntry):
                renderables.append(self._tool_renderable(entry))
            elif entry.role == "user":
                renderables.append(Text(f"› {entry.text}", style="cyan"))
            elif entry.role == "assistant":
                renderables.append(Markdown(entry.text))
            elif entry.role == "thinking":
                renderables.append(
                    Group(
                        Text("thinking", style="dim"),
                        Text(entry.text, style="dim", overflow="fold"),
                    )
                )
            elif entry.role == "notice":
                renderables.append(Text(entry.text, style="dim", overflow="fold"))
            else:
                renderables.append(Text(f"Error: {entry.text}", style="red", overflow="fold"))
        return Group(*renderables)

    def _visible_entries(self) -> list[_TextEntry | _ToolEntry]:
        return [
            entry
            for entry in self._entries
            if not (
                isinstance(entry, _TextEntry)
                and entry.role == "thinking"
                and not self.show_thinking
            )
        ]

    def _append_text(self, role: str, text: str) -> None:
        if self._entries and isinstance(self._entries[-1], _TextEntry):
            current = self._entries[-1]
            if current.role == role:
                current.text += text
                return
        self._entries.append(_TextEntry(role, text))

    def _add_tool_call(self, part: ToolCallPart) -> None:
        if part.tool_call_id in self._tools:
            return
        tool = _ToolEntry(
            tool_call_id=part.tool_call_id,
            name=part.tool_name,
            arguments=_value_text(part.args),
        )
        self._tools[part.tool_call_id] = tool
        self._entries.append(tool)

    def _add_tool_result(self, part: ToolReturnPart | RetryPromptPart) -> None:
        tool = self._tools.get(part.tool_call_id)
        if tool is None:
            name = part.tool_name or "tool"
            tool = _ToolEntry(part.tool_call_id, name, "")
            self._tools[part.tool_call_id] = tool
            self._entries.append(tool)
        tool.status = getattr(part, "outcome", "failed")
        tool.result = _value_text(part.content)

    def _tool_renderable(self, tool: _ToolEntry) -> Group:
        color = {
            "running": "yellow",
            "success": "green",
            "failed": "red",
            "denied": "red",
            "interrupted": "yellow",
        }.get(tool.status, "yellow")
        row = Text("  ")
        row.append(tool.name, style="bold")
        row.append(f"  {tool.status}", style=color)
        if tool.arguments:
            row.append(f"  {_one_line(tool.arguments, 120)}", style="dim")
        renderables: list[RenderableType] = [row]
        if self.show_tools and tool.result:
            renderables.append(
                Text(f"    {_bounded_detail(tool.result)}", style="dim", overflow="fold")
            )
        return Group(*renderables)


def _display_user_content(content: str | Sequence[UserContent]) -> str:
    if isinstance(content, str):
        return content
    for item in content:
        if isinstance(item, str):
            return item
        if isinstance(item, TextContent) and not item.content.startswith("<yolop-file "):
            return item.content
    return ""


def _value_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _one_line(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _bounded_detail(value: str) -> str:
    if len(value) <= _MAX_TOOL_DETAIL_CHARS:
        return value
    return value[:_MAX_TOOL_DETAIL_CHARS] + "\n… [tool output truncated for display]"
