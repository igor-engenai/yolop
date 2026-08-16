import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pytest import MonkeyPatch, raises

from yolop import Yolop


@dataclass(frozen=True)
class HostDeps:
    workspace: Path


def test_workspace_shell_requires_an_explicit_command_allowlist(tmp_path: Path) -> None:
    with raises(ValueError, match="shell requires a non-empty allowed_commands list"):
        Yolop().run(
            {"capabilities": [{"Workspace": {"shell": True}}]},
            "Run a command.",
            model=FunctionModel(stream_function=_unused_stream),
            deps=HostDeps(workspace=tmp_path),
            deps_type=HostDeps,
        )


async def _unused_stream(
    _messages: list[ModelMessage], _info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    yield "unused"


async def test_workspace_does_not_enable_shell_by_default(tmp_path: Path) -> None:
    async def respond(
        _messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        available_tools = {tool.name for tool in info.function_tools}
        assert "read_file" in available_tools
        assert "run_command" not in available_tools
        yield "File tools only"

    async with Yolop().run(
        {"capabilities": ["Workspace"]},
        "Inspect the workspace.",
        model=FunctionModel(stream_function=respond),
        deps=HostDeps(workspace=tmp_path),
        deps_type=HostDeps,
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "File tools only"


async def test_workspace_uses_host_path_for_files_and_shell(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-must-not-reach-shell")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key-must-not-reach-shell")

    command = (
        'python -c "import os; from pathlib import Path; '
        "print(Path('note.txt').read_text()); print(os.getenv('OPENAI_API_KEY')); "
        "print(os.getenv('AZURE_OPENAI_API_KEY'))\""
    )

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if len(tool_returns) == 0:
            available_tools = {tool.name for tool in info.function_tools}
            assert {"write_file", "read_file", "run_command"} <= available_tools
            yield {
                0: DeltaToolCall(
                    name="write_file",
                    json_args=json.dumps({"path": "note.txt", "content": "workspace works"}),
                    tool_call_id="write-note",
                )
            }
        elif len(tool_returns) == 1:
            yield {
                0: DeltaToolCall(
                    name="run_command",
                    json_args=json.dumps({"command": command}),
                    tool_call_id="read-note-with-shell",
                )
            }
        else:
            shell_output = str(tool_returns[-1].content)
            assert "workspace works" in shell_output
            assert "openai-key-must-not-reach-shell" not in shell_output
            assert "azure-key-must-not-reach-shell" not in shell_output
            yield "Workspace capability works"

    spec: dict[str, Any] = {
        "capabilities": [
            {
                "Workspace": {
                    "shell": True,
                    "allowed_commands": ["python"],
                }
            }
        ]
    }

    async with Yolop().run(
        spec,
        "Create a note, then read it with Python.",
        model=FunctionModel(stream_function=respond),
        deps=HostDeps(workspace=tmp_path),
        deps_type=HostDeps,
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Workspace capability works"
    assert (tmp_path / "note.txt").read_text() == "workspace works"
