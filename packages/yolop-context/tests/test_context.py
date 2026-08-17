from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.models.test import TestModel
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import TranscriptHandleProvider
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits
from yolop import ProviderCatalog
from yolop_runtime import ExecutionScope
from yolop_context import (
    Context,
    ContextConfigurationError,
    ContextResourceError,
    ScopedOverflowStore,
    TranscriptHandle,
)


@dataclass
class MemoryOverflowStore:
    values: dict[str, bytes] = field(default_factory=dict)

    async def write(self, key: str, data: bytes) -> str:
        self.values[key] = data
        return key

    async def read(self, handle: str) -> bytes:
        return self.values[handle]


def _context(spec: Context, *, deps: Any, run_id: str) -> RunContext[Any]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        run_id=run_id,
        capabilities={spec.get_serialization_name(): spec},
    )


def test_context_is_selected_by_the_normal_capability_catalog() -> None:
    class EntryPoint:
        name = "Context"
        value = "yolop_context:Context"
        dist = None

        @staticmethod
        def load() -> type[Context]:
            return Context

    catalog = ProviderCatalog.from_entry_points(capability_entry_points=[EntryPoint()])
    spec = AgentSpec.from_dict({"capabilities": [{"Context": {}}]})

    assert catalog.capability_types_for(spec) == (Context,)
    assert Context.from_spec() == Context()


def test_context_from_spec_rejects_runtime_objects_and_unknown_arguments() -> None:
    with pytest.raises(ContextConfigurationError, match="serialized"):
        Context.from_spec(store=MemoryOverflowStore())

    with pytest.raises(ContextConfigurationError, match="serialized"):
        Context.from_spec(overflow_threshold=object())


def test_context_requires_a_host_overflow_store_before_model_execution() -> None:
    context = Context()

    with pytest.raises(ContextResourceError, match="overflow_store"):
        asyncio_run(context.for_run(_context(context, deps=SimpleNamespace(), run_id="run")))


def test_scoped_overflow_handles_cannot_cross_sessions() -> None:
    store = MemoryOverflowStore()
    first = ScopedOverflowStore(store, namespace="tenant", session_id="session-a")
    second = ScopedOverflowStore(store, namespace="tenant", session_id="session-b")

    async def scenario() -> None:
        handle = await first.write("run/call", b"secret")
        assert await first.read(handle) == b"secret"
        with pytest.raises(PermissionError, match="scope"):
            await second.read(handle)

    asyncio_run(scenario())


def test_context_binds_tool_output_limits_and_transcript_handle() -> None:
    context = Context()
    store = MemoryOverflowStore()

    async def scenario() -> None:
        bound = await context.for_run(
            _context(
                context,
                deps=SimpleNamespace(
                    overflow_store=store,
                    scope=ExecutionScope(
                        "tenant",
                        "00000000-0000-4000-8000-000000000001",
                        "00000000-0000-4000-8000-000000000002",
                    ),
                ),
                run_id="run",
            )
        )
        assert isinstance(bound, CombinedCapability)
        assert any(isinstance(capability, ToolOutputLimits) for capability in bound.capabilities)
        transcript = next(
            capability
            for capability in bound.capabilities
            if isinstance(capability, TranscriptHandleProvider)
        )
        assert isinstance(transcript, TranscriptHandle)
        assert transcript.compaction_transcript_handle() == "run"

    asyncio_run(scenario())


def asyncio_run(awaitable: Any) -> Any:
    import asyncio

    return asyncio.run(awaitable)
