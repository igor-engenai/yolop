from __future__ import annotations

import base64
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import (
    TranscriptHandleProvider,
)
from pydantic_ai_harness.compaction import (
    WarnNearLimits as HarnessWarnNearLimits,
)
from pydantic_ai_harness.tool_output_limits import (
    Band,
    OverflowStore,
    Spill,
    ToolOutputLimits,
    Truncate,
)
from pydantic_core import to_json


class ContextConfigurationError(ValueError):
    """The serialized Context capability configuration is not safe or supported."""

    code = "context_configuration_error"


class ContextResourceError(UserError):
    """The host did not provide a resource required by the Context capability."""

    code = "context_resource_error"


class StuckLoopError(UserError):
    """A run repeated unproductive tool work beyond its configured limit."""

    code = "stuck_loop_detected"


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
class WarnNearLimits(HarnessWarnNearLimits[Any]):
    """Safely expose Harness limit warnings through AgentSpec."""

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> WarnNearLimits:
        if args or not _is_serialized_value(kwargs):
            raise ContextConfigurationError(
                "WarnNearLimits accepts serialized capability arguments only"
            )
        supported = {
            "max_iterations",
            "max_context_tokens",
            "max_context_fraction",
            "context_window",
            "fallback_context_window",
            "max_total_tokens",
            "warn_on",
            "warning_threshold",
            "critical_remaining_iterations",
        }
        unknown = sorted(set(kwargs) - supported)
        if unknown:
            raise ContextConfigurationError(
                f"WarnNearLimits arguments are unsupported: {', '.join(unknown)}"
            )
        warn_on = kwargs.get("warn_on")
        if warn_on is not None and (
            not isinstance(warn_on, list)
            or any(item not in {"iterations", "context_window", "total_tokens"} for item in warn_on)
        ):
            raise ContextConfigurationError("WarnNearLimits.warn_on contains an unsupported kind")
        return cls(**kwargs)


@dataclass
class StuckLoop(AbstractCapability[Any]):
    """Detect repeated calls or results without retaining their full payloads."""

    repeat_threshold: int = 3
    alternating_threshold: int = 2
    result_threshold: int = 3
    history_limit: int = 32
    ignored_tools: tuple[str, ...] = ()
    action: Literal["retry", "error"] = "retry"
    _call_history: deque[str] = field(init=False, repr=False)
    _result_history: deque[tuple[str, str]] = field(init=False, repr=False)

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> StuckLoop:
        if args or not _is_serialized_value(kwargs):
            raise ContextConfigurationError(
                "StuckLoop accepts serialized capability arguments only"
            )
        supported = {
            "repeat_threshold",
            "alternating_threshold",
            "result_threshold",
            "history_limit",
            "ignored_tools",
            "action",
        }
        unknown = sorted(set(kwargs) - supported)
        if unknown:
            raise ContextConfigurationError(
                f"StuckLoop arguments are unsupported: {', '.join(unknown)}"
            )
        return cls(**kwargs)

    def __post_init__(self) -> None:
        for name in (
            "repeat_threshold",
            "alternating_threshold",
            "result_threshold",
            "history_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContextConfigurationError(f"{name} must be a positive integer")
        if self.action not in {"retry", "error"}:
            raise ContextConfigurationError("StuckLoop.action must be 'retry' or 'error'")
        if not isinstance(self.ignored_tools, (list, tuple)) or not all(
            isinstance(tool, str) and tool for tool in self.ignored_tools
        ):
            raise ContextConfigurationError("StuckLoop.ignored_tools must contain tool names")
        self.ignored_tools = tuple(self.ignored_tools)
        self._call_history = deque(maxlen=self.history_limit)
        self._result_history = deque(maxlen=self.history_limit)

    @property
    def history_size(self) -> int:
        """Return the number of retained call signatures."""
        return len(self._call_history)

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        """Return a detector with fresh history for this Run."""
        return replace(self)

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: Any,
        tool_def: Any,
        args: Any,
    ) -> Any:
        del ctx
        tool_name = call.tool_name
        if tool_name in self.ignored_tools:
            return args
        signature = _fingerprint({"tool": tool_name, "args": args})
        self._call_history.append(signature)
        reason = self._call_reason()
        if reason is not None:
            self._raise_action(reason)
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: Any,
        tool_def: Any,
        args: Any,
        result: Any,
    ) -> Any:
        del ctx, args
        tool_name = call.tool_name
        if tool_name in self.ignored_tools:
            return result
        self._result_history.append((tool_name, _fingerprint(result)))
        if self._same_result_count() >= self.result_threshold:
            self._raise_action("repeated tool result")
        return result

    def _call_reason(self) -> str | None:
        if _tail_count(self._call_history) >= self.repeat_threshold:
            return "repeated tool call"
        width = self.alternating_threshold
        if len(self._call_history) < width * 2:
            return None
        tail = list(self._call_history)[-width * 2 :]
        if len(set(tail[:width])) == 1:
            return None
        if all(tail[index] == tail[index + width] for index in range(width)):
            return "alternating tool calls"
        return None

    def _same_result_count(self) -> int:
        if not self._result_history:
            return 0
        target = self._result_history[-1]
        count = 0
        for item in reversed(self._result_history):
            if item != target:
                break
            count += 1
        return count

    def _raise_action(self, reason: str) -> None:
        message = f"Stuck loop detected: {reason}; choose a different tool action."
        if self.action == "retry":
            raise ModelRetry(message)
        raise StuckLoopError(message)


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


def _tail_count(history: deque[str]) -> int:
    if not history:
        return 0
    target = history[-1]
    count = 0
    for item in reversed(history):
        if item != target:
            break
        count += 1
    return count


def _fingerprint(value: Any) -> str:
    try:
        encoded = to_json(value)
    except (TypeError, ValueError, OverflowError):
        encoded = (
            f"{type(value).__module__}:{type(value).__qualname__}:{repr(value)[:4096]}"
        ).encode()
    return sha256(encoded).hexdigest()


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
    "StuckLoop",
    "StuckLoopError",
    "WarnNearLimits",
    "ContextDeps",
    "ContextResourceError",
    "ContextScope",
    "ScopedOverflowStore",
    "TranscriptHandle",
]
