from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import TieredCompaction
from yolop_context import Compaction, TranscriptHandle


def _context(
    *,
    deps: Any = None,
    capabilities: dict[str, Any] | None = None,
) -> RunContext[Any]:
    return RunContext(
        deps=SimpleNamespace() if deps is None else deps,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        capabilities={} if capabilities is None else capabilities,
    )


def _tool_history() -> list[ModelRequest | ModelResponse]:
    return [
        ModelResponse(parts=[ToolCallPart("read_file", {"path": "a"}, tool_call_id="one")]),
        ModelRequest(parts=[ToolReturnPart("read_file", "old result", tool_call_id="one")]),
        ModelResponse(parts=[ToolCallPart("read_file", {"path": "a"}, tool_call_id="two")]),
        ModelRequest(parts=[ToolReturnPart("read_file", "new result", tool_call_id="two")]),
    ]


def test_compaction_accepts_only_safe_serialized_policy() -> None:
    configured = Compaction.from_spec(
        target_tokens=10,
        file_tools=["read_file"],
        include_summarizer=False,
    )
    assert configured.target_tokens == 10
    assert configured.file_tools == ("read_file",)

    with pytest.raises(ValueError, match="unsupported"):
        Compaction.from_spec(model="secret-model")


@pytest.mark.asyncio
async def test_cheap_compaction_clears_old_tool_results_without_summarizing() -> None:
    capability = Compaction(
        target_tokens=1,
        keep_tool_pairs=1,
        include_summarizer=False,
    )
    bound = await capability.for_run(_context())
    assert isinstance(bound, TieredCompaction)

    compacted = await bound.compact(_tool_history(), _context())
    assert _return_content(compacted[1]) == "[tool result cleared]"
    assert _return_content(compacted[3]) == "new result"


@pytest.mark.asyncio
async def test_file_read_deduplication_keeps_the_latest_result() -> None:
    capability = Compaction(
        target_tokens=1,
        file_tools=["read_file"],
        include_summarizer=False,
    )
    bound = await capability.for_run(_context())

    tiered = cast(TieredCompaction[Any], bound)
    compacted = await tiered.compact(_tool_history(), _context())
    assert _return_content(compacted[1]) == "[superseded file read]"
    assert _return_content(compacted[3]) == "new result"


@pytest.mark.asyncio
async def test_summarizer_is_lazy_incremental_and_receipted() -> None:
    prompts: list[str] = []

    def summarize(messages: list[Any], _info: Any) -> ModelResponse:
        prompts.append(str(messages[-1]))
        return ModelResponse(parts=[TextPart("updated summary")], model_name="claude:summary")

    summarizer = FunctionModel(function=summarize, model_name="claude:summary")
    scope = SimpleNamespace(namespace="tenant", session_id="session", run_id="run")
    deps = SimpleNamespace(scope=scope, summarizer_model=summarizer)
    context = _context(
        deps=deps,
        capabilities={"TranscriptHandle": TranscriptHandle("run")},
    )
    capability = Compaction(
        target_tokens=1,
        keep_tool_pairs=0,
        summarizer_keep_messages=1,
        bridge_prefix=True,
    )
    bound = await capability.for_run(context)
    assert isinstance(bound, CombinedCapability)
    tiered = cast(TieredCompaction[Any], bound.capabilities[0])
    history = [
        ModelRequest(parts=[UserPromptPart("old")]),
        ModelResponse(parts=[TextPart("old answer")], model_name="openai:gpt"),
        ModelRequest(parts=[UserPromptPart("current")]),
    ]

    first = await tiered.compact(history, context)
    assert len(prompts) == 1
    assert context.usage.requests == 1
    receipt = next(
        part
        for message in first
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
        and isinstance(part.content, list)
        and any(
            isinstance(item, TextContent)
            and item.metadata == "pydantic-ai-harness.compaction.receipt.v1"
            for item in part.content
        )
    )
    assert isinstance(receipt.content, list)
    assert "run" in str(receipt.content)
    assert "secondhand" in str(receipt.content)
    assert any(
        isinstance(message, ModelRequest)
        and any(
            isinstance(part, SystemPromptPart) and "different model" in part.content
            for part in message.parts
        )
        for message in first
    )

    await tiered.compact(
        [*first, ModelRequest(parts=[UserPromptPart("new")])],
        context,
    )
    assert len(prompts) == 2
    assert "previous-summary" in prompts[-1]


def _return_content(message: object) -> object:
    assert isinstance(message, ModelRequest)
    part = message.parts[0]
    assert isinstance(part, ToolReturnPart)
    return part.content


def test_canonical_turn_projection_drops_rewritten_prior_active_history() -> None:
    from yolop_runtime import canonical_turn_messages

    rewritten = [
        ModelRequest(parts=[UserPromptPart("compacted summary")]),
        ModelRequest(parts=[UserPromptPart("current prompt")]),
        ModelResponse(parts=[]),
    ]

    assert canonical_turn_messages(rewritten, "current prompt") == rewritten[1:]
