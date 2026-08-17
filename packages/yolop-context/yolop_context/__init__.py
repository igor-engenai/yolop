from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import TranscriptHandleProvider
from pydantic_ai_harness.tool_output_limits import (
    Band,
    OverflowStore,
    Spill,
    ToolOutputLimits,
    Truncate,
)


class ContextConfigurationError(ValueError):
    """The serialized Context capability configuration is not safe or supported."""

    code = "context_configuration_error"


class ContextResourceError(UserError):
    """The host did not provide a resource required by the Context capability."""

    code = "context_resource_error"


@runtime_checkable
class ContextScope(Protocol):
    """The durable scope identity required by host context resources."""

    namespace: str
    session_id: str
    run_id: str


class ContextDeps(Protocol):
    """Host dependencies required by the Context capability."""

    @property
    def overflow_store(self) -> OverflowStore: ...

    @property
    def scope(self) -> ContextScope: ...


@dataclass(frozen=True)
class ScopedOverflowStore:
    """Bind a Harness overflow store to one YoloP namespace and Session."""

    store: OverflowStore
    namespace: str
    session_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("Scoped overflow namespace must not be empty")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("Scoped overflow session_id must not be empty")

    async def write(self, key: str, data: bytes) -> str:
        backend_key = f"{self._prefix()}{key}"
        handle = await self.store.write(backend_key, data)
        if not isinstance(handle, str) or not handle:
            raise ContextResourceError("overflow_store.write must return a non-empty handle")
        return f"{self._prefix()}{handle}"

    async def read(self, handle: str) -> bytes:
        prefix = self._prefix()
        if not isinstance(handle, str) or not handle.startswith(prefix):
            raise PermissionError("Overflow handle belongs to a different scope")
        return await self.store.read(handle[len(prefix) :])

    def _prefix(self) -> str:
        namespace = base64.urlsafe_b64encode(self.namespace.encode()).decode().rstrip("=")
        return f"yolop-context-v1/{namespace}/{self.session_id}/"


@dataclass
class TranscriptHandle(AbstractCapability[Any], TranscriptHandleProvider):
    """Expose the durable YoloP Run ID to Harness compaction receipts."""

    handle: str

    def __post_init__(self) -> None:
        if not isinstance(self.handle, str) or not self.handle:
            raise ValueError("Transcript handle must not be empty")

    def compaction_transcript_handle(self) -> str:
        return self.handle


@dataclass
class Context(AbstractCapability[Any]):
    """Bind safe Harness context helpers to host-scoped YoloP resources."""

    overflow_threshold: int = 10_000
    preview_chars: int = 1_000
    truncate_chars: int = 4_000

    def __post_init__(self) -> None:
        for name in ("overflow_threshold", "preview_chars", "truncate_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContextConfigurationError(f"{name} must be a positive integer")

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> Context:
        """Construct only from bounded JSON-like AgentSpec arguments."""
        if args or not _is_serialized_value(kwargs):
            raise ContextConfigurationError("Context accepts serialized capability arguments only")
        supported = {"overflow_threshold", "preview_chars", "truncate_chars"}
        unknown = sorted(set(kwargs) - supported)
        if unknown:
            raise ContextConfigurationError(
                f"Context arguments are unsupported: {', '.join(unknown)}"
            )
        return cls(**kwargs)

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        """Resolve scoped storage and the durable transcript handle for this Run."""
        store = _overflow_store(ctx.deps)
        scope = _scope(ctx.deps)
        return CombinedCapability(
            [
                ToolOutputLimits(
                    bands=[
                        Band(
                            over=self.overflow_threshold,
                            action=Spill(
                                preview_chars=self.preview_chars,
                                then=Truncate(max_chars=self.truncate_chars),
                            ),
                        )
                    ],
                    store=ScopedOverflowStore(
                        store,
                        namespace=scope.namespace,
                        session_id=scope.session_id,
                    ),
                ),
                TranscriptHandle(handle=scope.run_id),
            ]
        )


def _overflow_store(deps: Any) -> OverflowStore:
    store = getattr(deps, "overflow_store", None)
    if (
        store is None
        or not callable(getattr(store, "write", None))
        or not callable(getattr(store, "read", None))
    ):
        raise ContextResourceError(
            "Context capability requires deps.overflow_store with async write/read methods"
        )
    return store


def _scope(deps: Any) -> ContextScope:
    scope = getattr(deps, "scope", None)
    if not isinstance(scope, ContextScope):
        raise ContextResourceError(
            "Context capability requires deps.scope with namespace, session_id, and run_id"
        )
    return scope


def _is_serialized_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_serialized_value(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_is_serialized_value(item) for item in value)
    return False


__all__ = [
    "Context",
    "ContextConfigurationError",
    "ContextDeps",
    "ContextResourceError",
    "ContextScope",
    "ScopedOverflowStore",
    "TranscriptHandle",
]
