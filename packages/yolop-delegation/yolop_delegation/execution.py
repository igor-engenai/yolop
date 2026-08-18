from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import RunContext
from yolop_runtime import (
    ExecutionPin,
    RunRelation,
    Runtime,
    RuntimeDeps,
    RuntimeRunSnapshot,
    RunUsage,
)

from . import (
    DelegateCatalog,
    DelegatePolicyError,
    DelegateResolution,
    ResolvedDelegate,
    bounded_idempotency_key,
)


class DelegateExecutionError(RuntimeError):
    """A child delegate failed and the host chose propagation."""

    code = "delegate_execution_failed"


@dataclass(frozen=True)
class DelegateRequest:
    """Bounded, fully resolved input for one child execution."""

    namespace: str
    parent_session_id: str
    parent_run_id: str
    root_run_id: str
    delegate: ResolvedDelegate
    task: str
    depth: int
    child_count: int
    idempotency_key: str


@dataclass(frozen=True)
class DelegateResult:
    """Bounded execution identity and result metadata."""

    status: str
    child_session_id: str
    child_run_id: str | None
    output: Any | None = None
    usage: RunUsage | None = None
    error_code: str | None = None


class DelegateExecutor(Protocol):
    """Host-owned executor used by the native delegation tool."""

    async def execute(self, request: DelegateRequest) -> DelegateResult: ...


@dataclass(frozen=True)
class DelegationHostPolicy:
    """Host bounds for model-created delegation input and output."""

    max_task_chars: int = 16_384
    max_output_bytes: int = 32 * 1024
    failure_mode: Literal["contain", "propagate"] = "contain"

    def __post_init__(self) -> None:
        if isinstance(self.max_task_chars, bool) or self.max_task_chars < 1:
            raise DelegatePolicyError("Delegation max_task_chars must be positive")
        if isinstance(self.max_output_bytes, bool) or self.max_output_bytes < 1:
            raise DelegatePolicyError("Delegation max_output_bytes must be positive")
        if self.failure_mode not in {"contain", "propagate"}:
            raise DelegatePolicyError("Delegation failure mode is unsupported")


class DelegationCapability(AbstractCapability[Any]):
    """Native Pydantic AI tool for one host-resolved delegate set."""

    def __init__(
        self,
        resolution: DelegateResolution,
        executor: DelegateExecutor,
        *,
        depth: int,
        child_count: int,
        host_policy: DelegationHostPolicy,
    ) -> None:
        self.id = "yolop.delegation"
        self.description = "Run one bounded task through a host-authorized child AgentSpec."
        self.defer_loading = False
        self.resolution = resolution
        self.executor = executor
        self.depth = depth
        self.child_count = child_count
        self.host_policy = host_policy
        self._child_count = child_count
        self._child_count_lock = asyncio.Lock()
        self._tools = Capability[Any](id=self.id, tools=())
        self._register_tools()

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def get_instructions(self) -> str:
        aliases = ", ".join(self.resolution.aliases)
        return (
            "Delegation runs an independent child agent. Select only an authorized alias, "
            f"from: {aliases}. Provide a bounded task and do not include credentials."
        )

    def get_toolset(self) -> Any:
        return self._tools.get_toolset()

    def _register_tools(self) -> None:
        @self._tools.tool
        async def delegate(ctx: RunContext[Any], alias: str, task: str) -> dict[str, Any]:
            selected = self.resolution.for_alias(alias)
            if not isinstance(task, str) or not task.strip():
                raise DelegatePolicyError("Delegate task must not be empty")
            if len(task) > self.host_policy.max_task_chars:
                raise DelegatePolicyError("Delegate task exceeds the host limit")
            async with self._child_count_lock:
                selected.validate_invocation(depth=self.depth, child_count=self._child_count)
                request_child_count = self._child_count
                self._child_count += 1
            if not isinstance(ctx.deps, RuntimeDeps):
                raise DelegatePolicyError("Delegation requires a Runtime execution scope")
            scope = ctx.deps.scope
            call_id = ctx.tool_call_id or task
            request = DelegateRequest(
                namespace=scope.namespace,
                parent_session_id=scope.session_id,
                parent_run_id=scope.run_id,
                root_run_id=scope.root_run_id or scope.run_id,
                delegate=selected,
                task=task,
                depth=self.depth,
                child_count=request_child_count,
                idempotency_key=bounded_idempotency_key("delegate", scope.run_id, call_id),
            )
            try:
                result = await self.executor.execute(request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self.host_policy.failure_mode == "propagate":
                    raise DelegateExecutionError("Delegate execution failed") from error
                return {"status": "failed", "error_code": "delegate_failed"}
            if result.status != "completed" and self.host_policy.failure_mode == "propagate":
                raise DelegateExecutionError("Delegate execution failed")
            return _result_payload(result, max_bytes=self.host_policy.max_output_bytes)


def build_delegation_capability(
    namespace: str,
    spec: AgentSpec | Mapping[str, Any],
    *,
    catalog: DelegateCatalog,
    executor: DelegateExecutor,
    depth: int = 0,
    child_count: int = 0,
    host_policy: DelegationHostPolicy | None = None,
    max_task_chars: int | None = None,
    max_output_bytes: int | None = None,
) -> DelegationCapability | None:
    """Resolve parent selections and build native delegation tools."""
    resolution = catalog.resolve_for_spec(namespace, spec)
    if not resolution.selected:
        return None
    policy = host_policy or DelegationHostPolicy(
        max_task_chars=16_384 if max_task_chars is None else max_task_chars,
        max_output_bytes=32 * 1024 if max_output_bytes is None else max_output_bytes,
    )
    if isinstance(depth, bool) or depth < 0:
        raise DelegatePolicyError("Delegation depth must be non-negative")
    if isinstance(child_count, bool) or child_count < 0:
        raise DelegatePolicyError("Delegation child count must be non-negative")
    return DelegationCapability(
        resolution,
        executor,
        depth=depth,
        child_count=child_count,
        host_policy=policy,
    )


class RuntimeDelegateExecutor:
    """Run a resolved delegate through fresh durable Runtime Sessions and Runs."""

    def __init__(
        self,
        runtime: Runtime[Any],
        *,
        catalog: DelegateCatalog,
        model_for_id: Callable[[str], Model | KnownModelName | str],
        deps_for_request: Callable[[DelegateRequest], tuple[Any, type[Any]]],
    ) -> None:
        self.runtime = runtime
        self.catalog = catalog
        self.model_for_id = model_for_id
        self.deps_for_request = deps_for_request
        self._execution_locks: dict[str, asyncio.Lock] = {}

    async def execute(self, request: DelegateRequest) -> DelegateResult:
        lock = self._execution_locks.setdefault(request.idempotency_key, asyncio.Lock())
        async with lock:
            selected = self.catalog.resolve_pin(request.namespace, request.delegate.pin)
            selected.validate_invocation(
                depth=request.depth,
                child_count=request.child_count,
            )
            existing = await _find_child_run(
                self.runtime,
                request.namespace,
                request.idempotency_key,
                parent_run_id=request.parent_run_id,
            )
            if existing is not None:
                return _result_from_run(existing)
            child_spec = selected.spec
            child_session = await self.runtime.create_session(
                request.namespace,
                spec=child_spec,
                model_id=selected.model_id,
            )
            deps, deps_type = self.deps_for_request(request)
            try:
                completion = await self.runtime.run(
                    request.namespace,
                    child_session.id,
                    request.task,
                    spec=child_spec,
                    model=self.model_for_id(selected.model_id),
                    model_id=selected.model_id,
                    execution_pin=ExecutionPin.from_spec(
                        child_spec,
                        model_id=selected.model_id,
                    ),
                    parent_run_id=request.parent_run_id,
                    relation=RunRelation.CHILD,
                    deps=deps,
                    deps_type=deps_type,
                    idempotency_key=request.idempotency_key,
                    initiator="delegate",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                failed = await _find_child_run(
                    self.runtime,
                    request.namespace,
                    request.idempotency_key,
                    parent_run_id=request.parent_run_id,
                )
                return DelegateResult(
                    status="failed" if failed is None else failed.status.value,
                    child_session_id=child_session.id,
                    child_run_id=None if failed is None else failed.id,
                    error_code=None if failed is None else failed.error_code or "delegate_failed",
                    usage=None if failed is None else failed.usage,
                )
            return DelegateResult(
                status=completion.run.status.value,
                child_session_id=child_session.id,
                child_run_id=completion.run.id,
                output=completion.run.output,
                usage=completion.run.usage,
                error_code=completion.run.error_code,
            )


async def _find_child_run(
    runtime: Runtime[Any],
    namespace: str,
    idempotency_key: str,
    *,
    parent_run_id: str,
) -> RuntimeRunSnapshot | None:
    runs = await runtime.list_runs(namespace)
    return next(
        (
            run
            for run in runs
            if run.idempotency_key == idempotency_key and run.parent_run_id == parent_run_id
        ),
        None,
    )


def _result_from_run(run: RuntimeRunSnapshot) -> DelegateResult:
    return DelegateResult(
        status=run.status.value,
        child_session_id=run.session_id,
        child_run_id=run.id,
        output=run.output,
        usage=run.usage,
        error_code=run.error_code,
    )


def _result_payload(result: DelegateResult, *, max_bytes: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": result.status,
        "child_session_id": result.child_session_id,
        "child_run_id": result.child_run_id,
    }
    if result.error_code is not None:
        payload["error_code"] = result.error_code
    if result.usage is not None:
        payload["usage"] = {
            "requests": result.usage.requests,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        }
    if result.output is None:
        return payload
    if isinstance(result.output, str):
        text = result.output
    else:
        try:
            text = json.dumps(result.output, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(result.output)
    encoded = text.encode()
    if len(encoded) > max_bytes:
        payload["output"] = encoded[:max_bytes].decode(errors="ignore")
        payload["output_truncated"] = True
    else:
        payload["output"] = text
        payload["output_truncated"] = False
    return payload


__all__ = [
    "DelegateExecutionError",
    "DelegateExecutor",
    "DelegateRequest",
    "DelegateResult",
    "DelegationCapability",
    "DelegationHostPolicy",
    "RuntimeDelegateExecutor",
    "build_delegation_capability",
]
