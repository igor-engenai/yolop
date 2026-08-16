import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import yolop_tui.app as tui_app
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pytest import MonkeyPatch
from textual.pilot import Pilot
from textual.widgets import Static
from yolop_tui import __file__ as tui_package_file
from yolop_tui import run_tui
from yolop_tui.textual_app import TextualTerminal
from yolop_workspace_session import WorkspaceRuntimeStore

from examples.coding_tui import SPEC_PATH, HostDeps


def _install_driver(monkeypatch: MonkeyPatch, prompt: str) -> None:
    async def drive(pilot: Pilot) -> None:
        status = pilot.app.query_one("#status", Static)
        await pilot.press(*prompt, "enter")
        async with asyncio.timeout(1):
            while "↑0 ↓0" in str(status.render()):
                await pilot.pause(0.01)
        await pilot.press("/", "q", "u", "i", "t", "enter")

    def terminal_factory(**kwargs) -> TextualTerminal:
        return TextualTerminal(**kwargs, auto_pilot=drive)

    monkeypatch.setattr(tui_app, "_TERMINAL_FACTORY", terminal_factory)


async def test_coding_tui_injects_workspace_dependencies(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    async def respond(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        assert {tool.name for tool in info.function_tools} >= {"read_file", "run_command"}
        yield "Coding TUI works"

    _install_driver(monkeypatch, "Inspect the project")
    spec = AgentSpec.from_file(SPEC_PATH)
    store = WorkspaceRuntimeStore(tmp_path)
    await run_tui(
        spec,
        store=store,
        namespace="coding",
        deps=HostDeps(workspace=tmp_path),
        deps_type=HostDeps,
        model=FunctionModel(stream_function=respond),
        model_id="test:model",
        cwd=tmp_path,
    )

    assert (tmp_path / ".yolop" / "runtime.db").is_file()


async def test_bundled_tui_agent_can_write_to_the_injected_workspace(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    async def respond(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            assert {tool.name for tool in info.function_tools} >= {"write_file", "run_command"}
            yield {
                0: DeltaToolCall(
                    name="write_file",
                    json_args=json.dumps(
                        {"path": "default-agent.txt", "content": "workspace enabled"}
                    ),
                    tool_call_id="write-default-agent",
                )
            }
        else:
            yield "Default Workspace works"

    _install_driver(monkeypatch, "Write the test file")
    spec = AgentSpec.from_file(Path(tui_package_file).parent / "agent_specs" / "coding.yaml")
    store = WorkspaceRuntimeStore(tmp_path)
    await run_tui(
        spec,
        store=store,
        namespace="default-coding",
        deps=HostDeps(workspace=tmp_path),
        deps_type=HostDeps,
        model=FunctionModel(stream_function=respond),
        model_id="test:model",
        cwd=tmp_path,
    )

    assert (tmp_path / "default-agent.txt").read_text() == "workspace enabled"
