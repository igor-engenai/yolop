import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from yolop_tui import run_tui
from yolop_workspace_session import WorkspaceRuntimeStore

from examples.coding_tui import SPEC_PATH, HostDeps


class CapturingOutput(DummyOutput):
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, data: str) -> None:
        self.writes.append(data)

    def write_raw(self, data: str) -> None:
        self.writes.append(data)

    def get_size(self) -> Size:
        return Size(rows=24, columns=80)

    @property
    def text(self) -> str:
        return "".join(self.writes)


async def test_coding_tui_injects_workspace_dependencies(tmp_path: Path) -> None:
    async def respond(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        assert {tool.name for tool in info.function_tools} >= {"read_file", "run_command"}
        yield "Coding TUI works"

    output = CapturingOutput()
    spec = AgentSpec.from_file(SPEC_PATH)
    store = WorkspaceRuntimeStore(tmp_path)
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            running = asyncio.create_task(
                run_tui(
                    spec,
                    store=store,
                    namespace="coding",
                    deps=HostDeps(workspace=tmp_path),
                    deps_type=HostDeps,
                    model=FunctionModel(stream_function=respond),
                    model_id="test:model",
                    cwd=tmp_path,
                )
            )
            async with asyncio.timeout(1):
                while "╭─ prompt" not in output.text:
                    await asyncio.sleep(0)
            pipe_input.send_text("Inspect the project\r")
            async with asyncio.timeout(1):
                while "Coding TUI works" not in output.text:
                    await asyncio.sleep(0)
            session_id = (await store.list_sessions("coding"))[0]
            async with asyncio.timeout(1):
                while not (await store.load_session("coding", session_id)).messages:
                    await asyncio.sleep(0)
            pipe_input.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=1)

    assert (tmp_path / ".yolop" / "runtime.db").is_file()
