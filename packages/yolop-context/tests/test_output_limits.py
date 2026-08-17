from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits as HarnessToolOutputLimits
from yolop_context import (
    ContextScope,
    OutputBand,
    ScopedOverflowStore,
    ToolOutputLimits,
    cleanup_session_artifacts,
)


@dataclass
class MemoryStore:
    values: dict[str, bytes] = field(default_factory=dict)
    fail_writes: bool = False

    async def write(self, key: str, data: bytes) -> str:
        if self.fail_writes:
            raise OSError("backend path must not leak")
        self.values[key] = data
        return key

    async def read(self, handle: str) -> bytes:
        return self.values[handle]


@dataclass
class RecordingRegistry:
    records: list[tuple[ContextScope, str, int]] = field(default_factory=list)
    deleted: list[tuple[str, str]] = field(default_factory=list)

    async def record_artifact(self, scope: ContextScope, handle: str, *, size: int) -> None:
        self.records.append((scope, handle, size))

    async def delete_session_artifacts(self, namespace: str, session_id: str) -> None:
        self.deleted.append((namespace, session_id))


def _scope() -> SimpleNamespace:
    return SimpleNamespace(
        namespace="tenant",
        session_id="session",
        run_id="run",
    )


def _context(store: MemoryStore, registry: RecordingRegistry | None = None) -> RunContext[Any]:
    return RunContext(
        deps=SimpleNamespace(overflow_store=store, scope=_scope(), artifact_registry=registry),
        model=TestModel(),
        usage=RunUsage(),
        run_id="provider-run",
    )


def _call() -> tuple[ToolCallPart, ToolDefinition]:
    return ToolCallPart(tool_name="read_file"), ToolDefinition(name="read_file")


@pytest.mark.asyncio
async def test_small_tool_output_passes_unchanged() -> None:
    capability = ToolOutputLimits(bands=[OutputBand(over=10, action="spill")])
    bound = await capability.for_run(_context(MemoryStore()))
    assert isinstance(bound, HarnessToolOutputLimits)
    call, definition = _call()

    result = await bound.after_tool_execute(
        _context(MemoryStore()),
        call=call,
        tool_def=definition,
        args={},
        result="small",
    )
    assert result == "small"


@pytest.mark.asyncio
async def test_large_text_spills_with_bounded_preview_and_records_ownership() -> None:
    store = MemoryStore()
    registry = RecordingRegistry()
    capability = ToolOutputLimits(
        bands=[OutputBand(over=5, action="spill", preview_chars=4)],
    )
    bound = await capability.for_run(_context(store, registry))
    call, definition = _call()

    result = await bound.after_tool_execute(
        _context(store, registry),
        call=call,
        tool_def=definition,
        args={},
        result="abcdefghij",
    )
    assert isinstance(result, str)
    assert "stored to handle" in result
    assert "abcdefghij" not in result
    assert len(store.values) == 1
    assert registry.records[0][2] == 10


@pytest.mark.asyncio
async def test_failed_spill_uses_safe_truncation_fallback_without_backend_detail() -> None:
    store = MemoryStore(fail_writes=True)
    capability = ToolOutputLimits(
        bands=[OutputBand(over=5, action="spill", fallback="truncate", max_chars=4)],
    )
    bound = await capability.for_run(_context(store))
    call, definition = _call()

    result = await bound.after_tool_execute(
        _context(store),
        call=call,
        tool_def=definition,
        args={},
        result="abcdefghij",
    )
    assert "path must not leak" not in str(result)
    assert "truncated" in str(result)


@pytest.mark.asyncio
async def test_structured_and_binary_returns_use_native_tool_output_envelopes() -> None:
    store = MemoryStore()
    capability = ToolOutputLimits(bands=[OutputBand(over=1, action="spill")])
    bound = await capability.for_run(_context(store))
    call, definition = _call()

    structured = await bound.after_tool_execute(
        _context(store),
        call=call,
        tool_def=definition,
        args={},
        result=ToolReturn(return_value={"answer": 42}, metadata={"keep": "yes"}),
    )
    assert isinstance(structured, ToolReturn)
    assert structured.metadata["keep"] == "yes"
    assert "overflow_handle" in structured.metadata

    binary = await bound.after_tool_execute(
        _context(store),
        call=call,
        tool_def=definition,
        args={},
        result=b"binary payload",
    )
    assert isinstance(binary, ToolReturn)


@pytest.mark.asyncio
async def test_session_cleanup_is_explicit_and_does_not_delete_completed_run_data() -> None:
    registry = RecordingRegistry()
    deps = SimpleNamespace(artifact_registry=registry)

    await cleanup_session_artifacts(deps, namespace="tenant", session_id="session")
    assert registry.deleted == [("tenant", "session")]


@pytest.mark.asyncio
async def test_scoped_store_records_owner_with_an_explicit_registry() -> None:
    store = MemoryStore()
    registry = RecordingRegistry()
    scoped = ScopedOverflowStore(
        store,
        namespace="tenant",
        session_id="session",
        run_id="run",
        registry=registry,
    )

    await scoped.write("call", b"payload")
    assert registry.records == [(_scope(), registry.records[0][1], 7)]
