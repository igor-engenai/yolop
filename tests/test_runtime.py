from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass

from pydantic_ai import AgentRunResultEvent, AgentSpec, AgentStreamEvent, RunContext, TemplateStr
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
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


async def test_runtime_accepts_native_structured_user_content() -> None:
    prompt = [
        TextContent(content="Explain the file"),
        TextContent(content='<yolop-file path="answer.py">\nANSWER = 42\n</yolop-file>'),
    ]

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        user_prompt = request.parts[-1]
        assert isinstance(user_prompt, UserPromptPart)
        assert user_prompt.content == prompt
        yield "It defines the answer."

    async with Yolop().run(
        AgentSpec(),
        prompt,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        async for _event in run:
            pass

    assert run.result is not None
    assert run.result.output == "It defines the answer."


async def test_execute_exposes_native_run_context_to_the_event_handler() -> None:
    event_kinds: list[str] = []
    contexts: list[RunContext[None]] = []

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "Handled"

    async def handle(
        context: RunContext[None],
        events: AsyncIterable[AgentStreamEvent],
    ) -> None:
        contexts.append(context)
        event_kinds.extend([event.event_kind async for event in events])

    result = await Yolop().execute(
        AgentSpec(),
        "Use the handler",
        event_stream_handler=handle,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    )

    assert result.output == "Handled"
    assert contexts
    assert "part_start" in event_kinds
