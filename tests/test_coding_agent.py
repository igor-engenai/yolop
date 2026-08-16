import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import AgentRunResultEvent, AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from yolop import Yolop, coding_agent_spec


@dataclass(frozen=True)
class HostDeps:
    workspace: Path


async def test_coding_agentspec_composes_workspace_shell_and_skills(tmp_path: Path) -> None:
    spec = coding_agent_spec()

    assert isinstance(spec, AgentSpec)
    assert spec.model == "openai:gpt-5.6-luna"
    assert spec.name == "coding"
    assert spec.description
    assert spec.model_settings == {"thinking": "minimal"}

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
            assert {"load_skill", "write_file", "run_command"} <= available_tools
            assert "coding agent" in (info.instructions or "")
            yield {
                0: DeltaToolCall(
                    name="load_skill",
                    json_args=json.dumps({"name": "tdd"}),
                    tool_call_id="load-tdd",
                )
            }
        elif len(tool_returns) == 1:
            assert "Write one failing test" in str(tool_returns[-1].content)
            yield {
                0: DeltaToolCall(
                    name="write_file",
                    json_args=json.dumps(
                        {"path": "result.txt", "content": "coding AgentSpec works"}
                    ),
                    tool_call_id="write-result",
                )
            }
        else:
            yield "Coding agent works"

    async with Yolop().run(
        spec,
        "Prove the coding agent works.",
        model=FunctionModel(stream_function=respond),
        deps=HostDeps(workspace=tmp_path),
        deps_type=HostDeps,
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Coding agent works"
    assert (tmp_path / "result.txt").read_text() == "coding AgentSpec works"
