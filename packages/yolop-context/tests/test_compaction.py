from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import TieredCompaction
from yolop_context import Compaction


def _context() -> RunContext[Any]:
    return RunContext(
        deps=SimpleNamespace(),
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


def _tool_history() -> list[ModelRequest | ModelResponse]:
    return [
        ModelResponse(parts=[ToolCallPart("read_file", {"path": "a"}, tool_call_id="one")]),
        ModelRequest(parts=[ToolReturnPart("read_file", "old result", tool_call_id="one")]),
        ModelResponse(parts=[ToolCallPart("read_file", {"path": "b"}, tool_call_id="two")]),
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
    old_return = compacted[1]
    assert isinstance(old_return, ModelRequest)
    assert old_return.parts[0].content == "[tool result cleared]"
    assert compacted[3].parts[0].content == "new result"


@pytest.mark.asyncio
async def test_file_read_deduplication_keeps_the_latest_result() -> None:
    capability = Compaction(
        target_tokens=1,
        file_tools=["read_file"],
        include_summarizer=False,
    )
    bound = await capability.for_run(_context())

    compacted = await bound.compact(_tool_history(), _context())
    old_return = compacted[1]
    assert isinstance(old_return, ModelRequest)
    assert old_return.parts[0].content == "[superseded file read]"
    assert compacted[3].parts[0].content == "new result"


def test_canonical_turn_projection_drops_rewritten_prior_active_history() -> None:
    from yolop_runtime import canonical_turn_messages

    rewritten = [
        ModelRequest(parts=[UserPromptPart("compacted summary")]),
        ModelRequest(parts=[UserPromptPart("current prompt")]),
        ModelResponse(parts=[]),
    ]

    assert canonical_turn_messages(rewritten, "current prompt") == rewritten[1:]
