from collections.abc import AsyncIterator

from pydantic_ai import AgentRunResultEvent, AgentSpec, DeferredToolRequests
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pytest import raises
from yolop_runtime import Runtime
from yolop_sqlite_session import SQLiteRuntimeStore

from yolop import CapabilityPolicyConflictError, ToolAuditRecord, ToolPolicy, Yolop


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
            AgentSpec(capabilities=[{"Thinking": {}}]),
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


async def test_host_policy_rewrites_tool_args_and_audits_without_payloads() -> None:
    calls: list[str] = []
    audit: list[ToolAuditRecord] = []
    tools = Capability(id="host-tools")

    @tools.tool_plain
    def record_tool(value: str) -> str:
        calls.append(value)
        return "recorded"

    policy = ToolPolicy(
        rewrite=lambda context: {"value": "rewritten"},
        audit=audit.append,
    )

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            yield {
                0: DeltaToolCall(
                    name="record_tool",
                    json_args='{"value":"secret-value"}',
                    tool_call_id="record-call",
                )
            }
        else:
            yield "done"

    async with Yolop().run(
        AgentSpec(),
        "Rewrite the call",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        mandatory_capabilities=[tools, policy],
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "done"
    assert calls == ["rewritten"]
    assert [record.phase for record in audit] == ["before_execute", "after_execute"]
    assert all(record.payload_digest for record in audit)
    assert all("secret-value" not in repr(record) for record in audit)


async def test_host_policy_can_suspend_for_native_approval() -> None:
    tools = Capability(id="host-tools")

    @tools.tool_plain(requires_approval=True)
    def protected_tool(value: str) -> str:
        return value

    async def respond(
        _messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        yield {
            0: DeltaToolCall(
                name="protected_tool",
                json_args='{"value":"needs-approval"}',
                tool_call_id="approval-call",
            )
        }

    async with Yolop().run(
        AgentSpec(),
        "Request approval",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        mandatory_capabilities=[tools],
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    pending = events[-1].result.output
    assert isinstance(pending, DeferredToolRequests)
    assert [call.tool_call_id for call in pending.approvals] == ["approval-call"]


async def test_runtime_preserves_native_approval_resume_operation(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    spec = AgentSpec(model="test:model")
    tools = Capability(id="host-tools")
    calls: list[str] = []

    @tools.tool_plain(requires_approval=True)
    def protected_tool(value: str) -> str:
        calls.append(value)
        return f"executed:{value}"

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            yield {
                0: DeltaToolCall(
                    name="protected_tool",
                    json_args='{"value":"approved-value"}',
                    tool_call_id="runtime-approval",
                )
            }
        else:
            assert tool_returns[-1].content == "executed:approved-value"
            yield "resumed"

    session = await runtime.create_session("test", spec=spec, model_id="test:model")
    pending = await runtime.run(
        "test",
        session.id,
        "request approval",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        output_type=[str, DeferredToolRequests],
        mandatory_capabilities=[tools],
        idempotency_key="approval",
    )

    assert isinstance(pending.run.output, dict)
    assert pending.run.output["approvals"][0]["tool_call_id"] == "runtime-approval"

    resumed = await runtime.resume_deferred_run(
        "test",
        session.id,
        pending.run.id,
        approvals={"runtime-approval": True},
        spec=spec,
        model=FunctionModel(stream_function=respond),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        mandatory_capabilities=[tools],
        idempotency_key="approval-resume",
    )

    assert resumed.run.output == "resumed"
    assert calls == ["approved-value"]


async def test_host_policy_denies_tool_call_with_safe_reason() -> None:
    calls: list[str] = []
    tools = Capability(id="host-tools")

    @tools.tool_plain
    def dangerous_tool(value: str) -> str:
        calls.append(value)
        return "executed"

    policy = ToolPolicy(deny=lambda _context: "workspace_command_denied")

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            yield {
                0: DeltaToolCall(
                    name="dangerous_tool",
                    json_args='{"value":"secret-value"}',
                    tool_call_id="danger-call",
                )
            }
        else:
            assert tool_returns[-1].outcome == "denied"
            assert tool_returns[-1].content == "workspace_command_denied"
            yield "denial observed"

    async with Yolop().run(
        AgentSpec(),
        "Deny the call",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        mandatory_capabilities=[tools, policy],
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "denial observed"
    assert calls == []
