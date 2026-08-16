from collections.abc import AsyncIterator
from dataclasses import dataclass

from pydantic_ai import AgentRunResultEvent, AgentSpec, TemplateStr
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from yolop import Yolop


@dataclass(frozen=True)
class RunDeps:
    name: str


async def test_agent_spec_runs_as_a_native_pydantic_event_stream() -> None:
    history = [
        ModelRequest(parts=[UserPromptPart("Earlier question")]),
        ModelResponse(parts=[TextPart("Earlier answer")]),
    ]

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        assert info.instructions == "Hello Igor"
        yield "AgentSpec works"

    spec = AgentSpec(instructions=TemplateStr("Hello {{ name }}"))
    runtime = Yolop()

    async with runtime.run(
        spec,
        "Try the agent",
        model=FunctionModel(stream_function=respond),
        deps=RunDeps(name="Igor"),
        deps_type=RunDeps,
        message_history=history,
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "AgentSpec works"
    assert run.result is final_event.result
    assert run.all_messages()[:2] == history
