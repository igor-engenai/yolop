from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from yolop import Yolop


async def test_bundled_skills_are_inactive_until_agent_spec_enables_them() -> None:
    async def respond(
        _messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        assert "tdd" not in (info.instructions or "")
        assert info.function_tools == []
        yield "No skills enabled"

    async with Yolop().run(
        {"instructions": "Base instructions only"},
        "Run without skills",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "No skills enabled"


async def test_agent_spec_enables_and_loads_a_bundled_skill() -> None:
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
        if not tool_returns:
            assert "tdd" in (info.instructions or "")
            assert [tool.name for tool in info.function_tools] == ["load_skill"]
            yield {
                0: DeltaToolCall(
                    name="load_skill",
                    json_args='{"name":"tdd"}',
                    tool_call_id="load-tdd",
                )
            }
        else:
            assert "Write one failing test" in str(tool_returns[-1].content)
            yield "Used the bundled skill"

    spec: dict[str, Any] = {
        "capabilities": [
            {"Skills": {"builtin": ["tdd"]}},
        ]
    }

    async with Yolop().run(
        spec,
        "Use test-driven development",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Used the bundled skill"


async def test_agent_spec_carries_an_inline_core_skill_snapshot() -> None:
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
        if not tool_returns:
            assert "company-python" in (info.instructions or "")
            assert "tdd" not in (info.instructions or "")
            yield {
                0: DeltaToolCall(
                    name="load_skill",
                    json_args='{"name":"company-python"}',
                    tool_call_id="load-company-python",
                )
            }
        else:
            assert "Use uv, ruff, and ty" in str(tool_returns[-1].content)
            yield "Used the Core skill"

    spec: dict[str, Any] = {
        "capabilities": [
            {
                "Skills": {
                    "custom": [
                        {
                            "name": "company-python",
                            "description": "Company Python conventions.",
                            "instructions": "Use uv, ruff, and ty.",
                        }
                    ]
                }
            }
        ]
    }

    async with Yolop().run(
        spec,
        "Follow company conventions",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Used the Core skill"
