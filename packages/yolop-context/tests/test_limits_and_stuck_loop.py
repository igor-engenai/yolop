from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from yolop_context import (
    StuckLoop,
    StuckLoopError,
    WarnNearLimits,
)


def _context() -> RunContext[Any]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def _call(name: str, args: dict[str, Any] | None = None) -> ToolCallPart:
    return ToolCallPart(tool_name=name, args=args or {})


def _definition(name: str) -> ToolDefinition:
    return ToolDefinition(name=name)


@pytest.mark.asyncio
async def test_warn_near_limits_injects_only_after_the_threshold() -> None:
    capability = WarnNearLimits(max_iterations=10, warning_threshold=0.5)
    ctx = _context()
    request = SimpleNamespace(messages=[], model=TestModel(), model_request_parameters=None)

    ctx.usage.requests = 4
    await capability.before_model_request(ctx, request)
    assert request.messages == []

    ctx.usage.requests = 5
    await capability.before_model_request(ctx, request)
    assert "[WarnNearLimits]" in request.messages[-1].parts[0].content


@pytest.mark.asyncio
async def test_stuck_loop_repeated_calls_trigger_native_retry_without_arguments() -> None:
    capability = StuckLoop(repeat_threshold=3, action="retry")
    ctx = _context()
    call = _call("search", {"secret": "do-not-leak"})
    definition = _definition("search")

    await capability.before_tool_execute(ctx, call=call, tool_def=definition, args=call.args)
    await capability.before_tool_execute(ctx, call=call, tool_def=definition, args=call.args)
    with pytest.raises(ModelRetry, match="repeated tool call") as error:
        await capability.before_tool_execute(ctx, call=call, tool_def=definition, args=call.args)
    assert "do-not-leak" not in str(error.value)


@pytest.mark.asyncio
async def test_stuck_loop_alternating_calls_trigger_correction() -> None:
    capability = StuckLoop(alternating_threshold=2, action="retry")
    ctx = _context()
    definitions = {name: _definition(name) for name in ("read", "write")}

    for name in ("read", "write", "read"):
        await capability.before_tool_execute(
            ctx,
            call=_call(name),
            tool_def=definitions[name],
            args={},
        )
    with pytest.raises(ModelRetry, match="alternating tool calls"):
        await capability.before_tool_execute(
            ctx,
            call=_call("write"),
            tool_def=definitions["write"],
            args={},
        )


@pytest.mark.asyncio
async def test_stuck_loop_repeated_results_trigger_terminal_error_without_result() -> None:
    capability = StuckLoop(result_threshold=2, action="error")
    ctx = _context()
    call = _call("poll")
    definition = _definition("poll")

    await capability.after_tool_execute(
        ctx, call=call, tool_def=definition, args={}, result={"secret": 1}
    )
    with pytest.raises(StuckLoopError, match="repeated tool result") as error:
        await capability.after_tool_execute(
            ctx,
            call=call,
            tool_def=definition,
            args={},
            result={"secret": 1},
        )
    assert "secret" not in str(error.value)
    assert error.value.code == "stuck_loop_detected"


@pytest.mark.asyncio
async def test_ignored_tools_do_not_trigger_and_each_run_gets_fresh_state() -> None:
    capability = StuckLoop(repeat_threshold=2, ignored_tools=("poll",))
    ctx = _context()
    call = _call("poll")
    definition = _definition("poll")

    for _ in range(3):
        await capability.before_tool_execute(ctx, call=call, tool_def=definition, args={})

    first = await capability.for_run(ctx)
    second = await capability.for_run(ctx)
    for bound in (first, second):
        await bound.before_tool_execute(
            ctx, call=_call("search"), tool_def=_definition("search"), args={}
        )
        await bound.before_tool_execute(
            ctx, call=_call("search"), tool_def=_definition("search"), args={}
        )


@pytest.mark.asyncio
async def test_stuck_loop_history_is_bounded() -> None:
    capability = StuckLoop(history_limit=3, repeat_threshold=99)
    ctx = _context()
    for index in range(20):
        name = f"tool-{index}"
        await capability.before_tool_execute(
            ctx,
            call=_call(name),
            tool_def=_definition(name),
            args={},
        )

    assert capability.history_size == 3
