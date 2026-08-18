from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from pydantic_ai import AgentRunResultEvent, AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pytest import raises
from test_delegation import definition
from yolop_delegation import (
    DelegateCatalog,
    DelegatePolicyError,
    DelegateRequest,
    DelegateResult,
    RuntimeDelegateExecutor,
    bounded_idempotency_key,
    build_delegation_capability,
)
from yolop_runtime import ExecutionScope, Runtime, RuntimeDeps, RunUsage
from yolop_sqlite_session import SQLiteRuntimeStore


class RecordingExecutor:
    def __init__(self, result: DelegateResult) -> None:
        self.result = result
        self.requests: list[Any] = []

    async def execute(self, request: Any) -> DelegateResult:
        self.requests.append(request)
        return self.result


async def test_delegate_tool_uses_only_the_parent_resolution_and_bounds_task() -> None:
    parent_spec = AgentSpec(
        metadata={
            "delegation": {"delegates": [{"alias": "research", "max_depth": 1, "max_children": 1}]}
        }
    )
    result = DelegateResult(
        status="completed",
        child_session_id="00000000-0000-4000-8000-000000000001",
        child_run_id="00000000-0000-4000-8000-000000000002",
        output="child output",
        usage=RunUsage(requests=1),
    )
    executor = RecordingExecutor(result)
    capability = build_delegation_capability(
        "tenant/acme",
        parent_spec,
        catalog=DelegateCatalog({"tenant/acme": [definition()]}),
        executor=executor,
        max_task_chars=20,
        max_output_bytes=100,
    )
    assert capability is not None

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        if not any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            assert [tool.name for tool in info.function_tools] == ["delegate"]
            yield {
                0: DeltaToolCall(
                    name="delegate",
                    json_args='{"alias":"research","task":"find facts"}',
                    tool_call_id="delegate-1",
                )
            }
        else:
            yield "parent done"

    from yolop import Yolop

    runtime_deps = RuntimeDeps(
        scope=ExecutionScope(
            namespace="tenant/acme",
            session_id="00000000-0000-4000-8000-000000000001",
            run_id="00000000-0000-4000-8000-000000000002",
        ),
        state=cast(Any, None),
        event_sink=None,
        follow_up_sink=None,
        host=None,
    )
    async with Yolop().run(
        parent_spec,
        "delegate work",
        model=FunctionModel(stream_function=respond),
        deps=runtime_deps,
        deps_type=RuntimeDeps,
        mandatory_capabilities=[capability],
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "parent done"
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.namespace == "tenant/acme"
    assert request.parent_run_id
    assert request.delegate.alias == "research"
    assert request.task == "find facts"


async def test_delegate_tool_enforces_total_child_limit() -> None:
    parent_spec = AgentSpec(
        metadata={
            "delegation": {
                "delegates": [{"alias": "research", "max_children": 1}]
            }
        }
    )
    executor = RecordingExecutor(
        DelegateResult(
            status="completed",
            child_session_id="00000000-0000-4000-8000-000000000001",
            child_run_id="00000000-0000-4000-8000-000000000002",
            output="child output",
        )
    )
    capability = build_delegation_capability(
        "tenant/acme",
        parent_spec,
        catalog=DelegateCatalog({"tenant/acme": [definition()]}),
        executor=executor,
    )
    assert capability is not None

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if len(returns) < 2:
            yield {
                0: DeltaToolCall(
                    name="delegate",
                    json_args='{"alias":"research","task":"find facts"}',
                    tool_call_id=f"delegate-{len(returns) + 1}",
                )
            }
        else:
            yield "done"

    from yolop import Yolop

    runtime_deps = RuntimeDeps(
        scope=ExecutionScope(
            namespace="tenant/acme",
            session_id="00000000-0000-4000-8000-000000000001",
            run_id="00000000-0000-4000-8000-000000000002",
        ),
        state=cast(Any, None),
        event_sink=None,
        follow_up_sink=None,
        host=None,
    )
    with raises(DelegatePolicyError, match="maximum children"):
        async with Yolop().run(
            parent_spec,
            "delegate work",
            model=FunctionModel(stream_function=respond),
            deps=runtime_deps,
            deps_type=RuntimeDeps,
            mandatory_capabilities=[capability],
        ) as run:
            _ = [event async for event in run]

    assert len(executor.requests) == 1


async def test_delegate_tool_bounds_large_child_output() -> None:
    parent_spec = AgentSpec(metadata={"delegation": {"delegates": [{"alias": "research"}]}})
    executor = RecordingExecutor(
        DelegateResult(
            status="completed",
            child_session_id="00000000-0000-4000-8000-000000000001",
            child_run_id="00000000-0000-4000-8000-000000000002",
            output="0123456789",
        )
    )
    capability = build_delegation_capability(
        "tenant/acme",
        parent_spec,
        catalog=DelegateCatalog({"tenant/acme": [definition()]}),
        executor=executor,
        max_output_bytes=4,
    )
    assert capability is not None
    observed: list[str] = []

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            yield {
                0: DeltaToolCall(
                    name="delegate",
                    json_args='{"alias":"research","task":"find facts"}',
                    tool_call_id="delegate-1",
                )
            }
        else:
            observed.append(str(returns[-1].content))
            yield "done"

    from yolop import Yolop

    runtime_deps = RuntimeDeps(
        scope=ExecutionScope(
            namespace="tenant/acme",
            session_id="00000000-0000-4000-8000-000000000001",
            run_id="00000000-0000-4000-8000-000000000002",
        ),
        state=cast(Any, None),
        event_sink=None,
        follow_up_sink=None,
        host=None,
    )
    async with Yolop().run(
        parent_spec,
        "delegate work",
        model=FunctionModel(stream_function=respond),
        deps=runtime_deps,
        deps_type=RuntimeDeps,
        mandatory_capabilities=[capability],
    ) as run:
        _ = [event async for event in run]

    assert len(observed) == 1
    assert "output_truncated" in observed[0]
    assert "0123" in observed[0]
    assert "0123456789" not in observed[0]


async def test_runtime_delegate_executor_creates_a_durable_child_session_and_run(
    tmp_path: Path,
) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    parent_spec = AgentSpec(
        model="parent:model",
        metadata={"delegation": {"delegates": [{"alias": "research"}]}},
    )
    catalog = DelegateCatalog(
        {
            "tenant/acme": [
                definition(
                    model_id="child:model",
                    max_depth=1,
                    max_children=1,
                )
            ]
        }
    )

    async def parent_respond(
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
                    name="delegate",
                    json_args='{"alias":"research","task":"research task"}',
                    tool_call_id="delegate-1",
                )
            }
        else:
            yield "parent received child result"

    async def child_respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "child result"

    parent_model = FunctionModel(stream_function=parent_respond)
    child_model = FunctionModel(stream_function=child_respond)
    executor = RuntimeDelegateExecutor(
        runtime,
        model_for_id=lambda model_id: child_model if model_id == "child:model" else model_id,
        deps_for_request=lambda _request: (None, type(None)),
        catalog=catalog,
    )
    capability = build_delegation_capability(
        "tenant/acme",
        parent_spec,
        catalog=catalog,
        executor=executor,
    )
    assert capability is not None
    parent = await runtime.create_session(
        "tenant/acme",
        spec=parent_spec,
        model_id="parent:model",
    )

    completion = await runtime.run(
        "tenant/acme",
        parent.id,
        "delegate work",
        spec=parent_spec,
        model=parent_model,
        model_id="parent:model",
        deps=None,
        deps_type=type(None),
        mandatory_capabilities=[capability],
        idempotency_key="parent-1",
    )

    assert completion.run.output == "parent received child result"
    child_sessions = [
        session_id
        for session_id in await runtime.list_sessions("tenant/acme")
        if session_id != parent.id
    ]
    assert len(child_sessions) == 1
    child_session = await runtime.load_session("tenant/acme", child_sessions[0])
    child_runs = await runtime.list_runs("tenant/acme", session_id=child_session.id)
    assert len(child_runs) == 1
    child_run = child_runs[0]
    assert child_session.pin.model_id == "child:model"
    assert child_run.session_id == child_session.id
    assert child_run.parent_run_id == completion.run.id
    assert (
        child_run.root_run_id == completion.run.root_run_id
        or child_run.root_run_id == completion.run.id
    )
    assert child_run.output == "child result"

    request = DelegateRequest(
        namespace="tenant/acme",
        parent_session_id=parent.id,
        parent_run_id=completion.run.id,
        root_run_id=completion.run.root_run_id or completion.run.id,
        delegate=catalog.resolve("tenant/acme", "research"),
        task="research task",
        depth=0,
        child_count=0,
        idempotency_key=bounded_idempotency_key("delegate", completion.run.id, "delegate-1"),
    )
    repeated = await executor.execute(request)
    assert repeated.child_session_id == child_session.id
    assert repeated.child_run_id == child_run.id
    assert len(await runtime.list_sessions("tenant/acme")) == 2
