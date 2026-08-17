from __future__ import annotations

import base64
import binascii
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, infer_model
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    DeduplicateFileReads,
    SummarizingCompaction,
    SupportsFocus,
    TieredCompaction,
    TranscriptHandleProvider,
)
from pydantic_ai_harness.compaction import (
    WarnNearLimits as HarnessWarnNearLimits,
)
from pydantic_ai_harness.tool_output_limits import (
    Band,
    OverflowStore,
    Passthrough,
    Spill,
    Summarize,
    Truncate,
    TruncationStrategy,
)
from pydantic_ai_harness.tool_output_limits import (
    ToolOutputLimits as HarnessToolOutputLimits,
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

    @property
    def namespace(self) -> str: ...

    @property
    def session_id(self) -> str: ...

    @property
    def run_id(self) -> str: ...


class ContextDeps(Protocol):
    """Host dependencies required by the Context capability."""

    @property
    def overflow_store(self) -> OverflowStore: ...

    @property
    def scope(self) -> ContextScope: ...

    @property
    def artifact_registry(self) -> ArtifactRegistry | None: ...


@runtime_checkable
class CompactionDeps(Protocol):
    """Optional host-selected resources for active-history compaction."""

    @property
    def scope(self) -> ContextScope: ...

    @property
    def summarizer_model(self) -> Model | str | None: ...


@runtime_checkable
class ArtifactRegistry(Protocol):
    """Host-owned artifact metadata and session cleanup boundary."""

    async def record_artifact(
        self,
        scope: ContextScope,
        handle: str,
        *,
        size: int,
    ) -> None: ...

    async def delete_session_artifacts(self, namespace: str, session_id: str) -> None: ...

    async def retain_run_artifacts(self, scope: ContextScope) -> None: ...


@dataclass(frozen=True)
class ScopedOverflowStore:
    """Bind a Harness overflow store to one YoloP namespace and Session."""

    store: OverflowStore
    namespace: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    registry: ArtifactRegistry | None = None
    scope: ContextScope | None = None

    def __post_init__(self) -> None:
        if self.scope is not None:
            object.__setattr__(self, "namespace", self.scope.namespace)
            object.__setattr__(self, "session_id", self.scope.session_id)
            object.__setattr__(self, "run_id", self.scope.run_id)
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("Scoped overflow namespace must not be empty")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("Scoped overflow session_id must not be empty")
        if self.registry is not None and (not isinstance(self.run_id, str) or not self.run_id):
            raise ContextResourceError("Artifact registry requires a durable Run ID")

    async def write(self, key: str, data: bytes) -> str:
        backend_key = f"{self._prefix()}{key}"
        handle = await self.store.write(backend_key, data)
        if not isinstance(handle, str) or not handle:
            raise ContextResourceError("overflow_store.write must return a non-empty handle")
        public_handle = self._pack(handle)
        if self.registry is not None:
            await self.registry.record_artifact(
                self._scope(),
                public_handle,
                size=len(data),
            )
        return public_handle

    async def read(self, handle: str) -> bytes:
        prefix = self._prefix()
        if not isinstance(handle, str) or not handle.startswith(prefix):
            raise PermissionError("Overflow handle belongs to a different scope")
        encoded = handle[len(prefix) :]
        try:
            padding = "=" * (-len(encoded) % 4)
            backend_handle = base64.urlsafe_b64decode(encoded + padding).decode()
        except (UnicodeDecodeError, ValueError, binascii.Error) as error:
            raise PermissionError("Overflow handle is invalid") from error
        return await self.store.read(backend_handle)

    def _pack(self, handle: str) -> str:
        encoded = base64.urlsafe_b64encode(handle.encode()).decode().rstrip("=")
        return f"{self._prefix()}{encoded}"

    def _prefix(self) -> str:
        assert self.namespace is not None
        assert self.session_id is not None
        namespace = base64.urlsafe_b64encode(self.namespace.encode()).decode().rstrip("=")
        return f"yolop-context-v1/{namespace}/{self.session_id}/"

    def _scope(self) -> ContextScope:
        if self.scope is not None:
            return self.scope
        assert self.namespace is not None
        assert self.session_id is not None
        assert self.run_id is not None
        return _BoundScope(self.namespace, self.session_id, self.run_id)


@dataclass(frozen=True)
class _BoundScope:
    namespace: str
    session_id: str
    run_id: str


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


@dataclass(frozen=True)
class OutputBand:
    """Serialized policy for one bounded tool-output action."""

    over: int
    action: Literal["truncate", "spill", "summarize"]
    max_chars: int = 4_000
    preview_chars: int = 1_000
    strategy: Literal["head", "tail", "head_tail"] = "head_tail"
    fallback: Literal["truncate", "spill", "summarize", "passthrough"] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.over, bool) or not isinstance(self.over, int) or self.over < 0:
            raise ContextConfigurationError("OutputBand.over must be a non-negative integer")
        for name in ("max_chars", "preview_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContextConfigurationError(f"OutputBand.{name} must be positive")
        if self.action not in {"truncate", "spill", "summarize"}:
            raise ContextConfigurationError("OutputBand.action is unsupported")
        if self.strategy not in {"head", "tail", "head_tail"}:
            raise ContextConfigurationError("OutputBand.strategy is unsupported")
        if self.fallback == self.action:
            raise ContextConfigurationError("OutputBand fallback must differ from its action")


@dataclass
class ToolOutputLimits(AbstractCapability[Any]):
    """Safely adapt Harness ToolOutputLimits to a host-scoped overflow store."""

    bands: Sequence[OutputBand] = field(
        default_factory=lambda: (OutputBand(over=10_000, action="spill", fallback="truncate"),)
    )
    over_tokens: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.bands, (list, tuple)) or not self.bands:
            raise ContextConfigurationError("ToolOutputLimits.bands must not be empty")
        normalized: list[OutputBand] = []
        for band in self.bands:
            if isinstance(band, OutputBand):
                normalized.append(band)
            elif isinstance(band, Mapping):
                try:
                    normalized.append(OutputBand(**band))
                except (TypeError, ValueError) as error:
                    raise ContextConfigurationError(
                        "ToolOutputLimits contains an invalid band"
                    ) from error
            else:
                raise ContextConfigurationError("ToolOutputLimits.bands must contain objects")
        self.bands = tuple(normalized)
        if not isinstance(self.over_tokens, bool):
            raise ContextConfigurationError("ToolOutputLimits.over_tokens must be a boolean")

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> ToolOutputLimits:
        if args or not _is_serialized_value(kwargs):
            raise ContextConfigurationError(
                "ToolOutputLimits accepts serialized capability arguments only"
            )
        supported = {"bands", "over_tokens"}
        unknown = sorted(set(kwargs) - supported)
        if unknown:
            raise ContextConfigurationError(
                f"ToolOutputLimits arguments are unsupported: {', '.join(unknown)}"
            )
        return cls(**kwargs)

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        store = _overflow_store(ctx.deps)
        scope = _scope(ctx.deps)
        registry = _artifact_registry(ctx.deps)
        return HarnessToolOutputLimits(
            bands=[_harness_band(band) for band in self.bands],
            over_tokens=self.over_tokens,
            store=ScopedOverflowStore(store, registry=registry, scope=scope),
        )


@dataclass
class Compaction(AbstractCapability[Any]):
    """Build cheap-to-expensive Harness compaction tiers from safe policy data."""

    target_tokens: int = 10_000
    keep_tool_pairs: int = 3
    file_tools: Sequence[str] = ()
    include_summarizer: bool = True
    summarizer_keep_messages: int = 20
    receipts: bool = True
    incremental: bool = True
    bridge_prefix: bool = False

    def __post_init__(self) -> None:
        for name in ("target_tokens", "keep_tool_pairs", "summarizer_keep_messages"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContextConfigurationError(f"Compaction.{name} must be non-negative")
        if self.target_tokens < 1:
            raise ContextConfigurationError("Compaction.target_tokens must be positive")
        if not isinstance(self.file_tools, (list, tuple)) or not all(
            isinstance(name, str) and name for name in self.file_tools
        ):
            raise ContextConfigurationError("Compaction.file_tools must contain tool names")
        if not all(
            isinstance(value, bool)
            for value in (
                self.include_summarizer,
                self.receipts,
                self.incremental,
                self.bridge_prefix,
            )
        ):
            raise ContextConfigurationError("Compaction flags must be booleans")
        self.file_tools = tuple(self.file_tools)

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> Compaction:
        if args or not _is_serialized_value(kwargs):
            raise ContextConfigurationError(
                "Compaction accepts serialized capability arguments only"
            )
        supported = {
            "target_tokens",
            "keep_tool_pairs",
            "file_tools",
            "include_summarizer",
            "summarizer_keep_messages",
            "receipts",
            "incremental",
            "bridge_prefix",
        }
        unknown = sorted(set(kwargs) - supported)
        if unknown:
            raise ContextConfigurationError(
                f"Compaction arguments are unsupported: {', '.join(unknown)}"
            )
        return cls(**kwargs)

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        tiers: list[Any] = [
            ClearToolResults(max_tokens=1, keep_pairs=self.keep_tool_pairs),
        ]
        if self.file_tools:
            tiers.append(
                DeduplicateFileReads(
                    file_key=_file_key_for(self.file_tools),
                )
            )
        if self.include_summarizer:
            summarizer_model = getattr(ctx.deps, "summarizer_model", None)
            tiers.append(
                SummarizingCompaction(
                    model=summarizer_model,
                    max_messages=1,
                    keep_messages=self.summarizer_keep_messages,
                    receipts=self.receipts,
                    incremental=self.incremental,
                    bridge_prefix=self.bridge_prefix,
                )
            )
        tiered = TieredCompaction(tiers=tiers, target_tokens=self.target_tokens)
        scope = getattr(ctx.deps, "scope", None)
        if isinstance(scope, ContextScope):
            return CombinedCapability([tiered, TranscriptHandle(handle=scope.run_id)])
        return tiered

    async def compact(
        self,
        messages: Sequence[ModelMessage],
        *,
        focus: str | None,
        model: Model | str,
        deps: Any,
        scope: ContextScope,
    ) -> list[ModelMessage]:
        """Compact active messages using the same strategy as automatic runs."""
        resolved_model = infer_model(model) if isinstance(model, str) else model
        transcript = TranscriptHandle(handle=scope.run_id)
        capabilities: dict[str, AbstractCapability[Any]] = {"TranscriptHandle": transcript}
        context = RunContext(
            deps=deps,
            model=resolved_model,
            usage=RunUsage(),
            run_id=scope.run_id,
            capabilities=capabilities,
        )
        bound = await self.for_run(context)
        if isinstance(bound, CombinedCapability):
            strategy = next(
                capability
                for capability in bound.capabilities
                if isinstance(capability, TieredCompaction)
            )
        else:
            if not isinstance(bound, TieredCompaction):
                raise ContextConfigurationError("Compaction did not build a tiered strategy")
            strategy = bound
        if focus is not None and isinstance(strategy, SupportsFocus):
            strategy = strategy.with_focus(focus)
        return await strategy.compact(list(messages), context)


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
        output = await ToolOutputLimits(
            bands=[
                OutputBand(
                    over=self.overflow_threshold,
                    action="spill",
                    preview_chars=self.preview_chars,
                    max_chars=self.truncate_chars,
                    fallback="truncate",
                )
            ]
        ).for_run(ctx)
        scope = _scope(ctx.deps)
        return CombinedCapability([output, TranscriptHandle(handle=scope.run_id)])


def _file_key_for(tool_names: Sequence[str]) -> Any:
    allowed = frozenset(tool_names)

    def file_key(call: Any) -> str | None:
        if call.tool_name not in allowed:
            return None
        try:
            args = call.args_as_dict()
        except (TypeError, ValueError):
            return None
        path = args.get("path") if isinstance(args, Mapping) else None
        return path if isinstance(path, str) and path else None

    return file_key


def _harness_band(band: OutputBand) -> Band:
    strategy = TruncationStrategy(band.strategy)
    fallback = _harness_action(band.fallback, band) if band.fallback is not None else None
    if band.action == "truncate":
        action: Any = Truncate(max_chars=band.max_chars, strategy=strategy, then=fallback)
    elif band.action == "spill":
        action = Spill(preview_chars=band.preview_chars, then=fallback)
    else:
        action = Summarize(then=fallback)
    return Band(over=band.over, action=action)


def _harness_action(
    action_name: Literal["truncate", "spill", "summarize", "passthrough"],
    band: OutputBand,
) -> Any:
    if action_name == "passthrough":
        return Passthrough()
    if action_name == "truncate":
        return Truncate(max_chars=band.max_chars, strategy=TruncationStrategy(band.strategy))
    if action_name == "spill":
        return Spill(preview_chars=band.preview_chars)
    return Summarize()


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


def _artifact_registry(deps: Any) -> ArtifactRegistry | None:
    registry = getattr(deps, "artifact_registry", None)
    if registry is None:
        return None
    if not callable(getattr(registry, "record_artifact", None)):
        raise ContextResourceError("Context capability requires artifact_registry.record_artifact")
    return registry


async def retain_run_artifacts(deps: Any, scope: ContextScope) -> None:
    """Mark completed-run artifacts for host retention instead of session cleanup."""
    registry = getattr(deps, "artifact_registry", None)
    if registry is None:
        return
    retain = getattr(registry, "retain_run_artifacts", None)
    if not callable(retain):
        raise ContextResourceError("Context artifact registry lacks retain_run_artifacts")
    await retain(scope)


async def cleanup_session_artifacts(
    deps: Any,
    *,
    namespace: str,
    session_id: str,
) -> None:
    """Delete artifacts owned by a Session through an explicit host registry."""
    registry = getattr(deps, "artifact_registry", None)
    if registry is None:
        return
    delete = getattr(registry, "delete_session_artifacts", None)
    if not callable(delete):
        raise ContextResourceError("Context artifact registry lacks delete_session_artifacts")
    await delete(namespace, session_id)


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
    "ArtifactRegistry",
    "Compaction",
    "CompactionDeps",
    "Context",
    "ContextConfigurationError",
    "StuckLoop",
    "StuckLoopError",
    "WarnNearLimits",
    "ContextDeps",
    "ContextResourceError",
    "ContextScope",
    "OutputBand",
    "ScopedOverflowStore",
    "ToolOutputLimits",
    "TranscriptHandle",
    "cleanup_session_artifacts",
    "retain_run_artifacts",
]
