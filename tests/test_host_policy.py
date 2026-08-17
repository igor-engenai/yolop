from collections.abc import AsyncIterator

from pydantic_ai import AgentRunResultEvent, AgentSpec
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pytest import raises

from yolop import CapabilityPolicyConflictError, Yolop


def test_host_capability_manifest_separates_spec_and_enforced_names() -> None:
    mandatory = Capability(id="host-required")

    resolution = Yolop().resolve_capabilities(
        AgentSpec(),
        mandatory_capabilities=[mandatory],
    )

    assert resolution.selected == ()
    assert resolution.enforced == ("host-required",)


def test_agent_spec_cannot_collide_with_a_mandatory_capability() -> None:
    with raises(CapabilityPolicyConflictError, match="Thinking"):
        Yolop().resolve_capabilities(
            AgentSpec(capabilities=["Thinking"]),
            mandatory_capabilities=[Capability(id="Thinking")],
        )


async def test_host_mandatory_capability_runs_when_spec_omits_it() -> None:
    calls: list[str] = []
    mandatory = Capability(id="host-required")

    @mandatory.tool_plain
    def required_tool(value: str) -> str:
        calls.append(value)
        return f"accepted:{value}"

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
            assert [tool.name for tool in info.function_tools] == ["required_tool"]
            yield {
                0: DeltaToolCall(
                    name="required_tool",
                    json_args='{"value":"from-host"}',
                    tool_call_id="required-call",
                )
            }
        else:
            assert tool_returns[-1].content == "accepted:from-host"
            yield "host capability executed"

    async with Yolop().run(
        AgentSpec(),
        "Use the host capability",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        mandatory_capabilities=[mandatory],
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "host capability executed"
    assert calls == ["from-host"]
