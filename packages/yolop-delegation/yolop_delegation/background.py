"""Durable background child supervision above the YoloP Runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import AgentSpec
from pydantic_ai.models import KnownModelName, Model
from yolop_runtime import (
    ExecutionPin,
    RunRelation,
    RunStatus,
    Runtime,
    ScopedStateContext,
    ensure_session_pin,
)

from . import (
    DelegateCatalog,
    DelegatePin,
    DelegatePolicyError,
    DelegateRequest,
)

_BACKGROUND_OWNER = "yolop.delegation.background"
_BACKGROUND_STATE_KIND = "tasks"
_BACKGROUND_SCHEMA_VERSION = 1


class BackgroundTaskConflictError(ValueError):
    """A background operation key has different durable input."""

    code = "background_task_conflict"


class BackgroundTaskNotFoundError(ValueError):
    """A background operation handle is not known in its parent Session."""

    code = "background_task_not_found"


class BackgroundTaskNotActiveError(RuntimeError):
    """A background child has no live worker steering sink."""

    code = "background_task_not_active"


class BackgroundTaskStatus(StrEnum):
    """Tool-facing projection of the canonical Runtime Run status."""

    ACCEPTED = RunStatus.ACCEPTED.value
    RUNNING = RunStatus.RUNNING.value
    COMPLETED = RunStatus.COMPLETED.value
    FAILED = RunStatus.FAILED.value
    INTERRUPTED = RunStatus.INTERRUPTED.value


class _BackgroundTaskRecord(BaseModel):
    """Durable tool handle; Run status remains the source of truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_key: str = Field(min_length=1, max_length=255)
    namespace: str
    parent_session_id: str
    parent_run_id: str
    child_session_id: str
    child_run_id: str
    task: str = Field(min_length=1, max_length=16_384)
    pin: DelegatePin
    depth: int = Field(ge=0)
    child_count: int = Field(ge=0)


@dataclass(frozen=True)
class BackgroundTaskHandle:
    """Durable child identity and a bounded current Run projection."""

    namespace: str
    operation_key: str
    parent_session_id: str
    parent_run_id: str
    child_session_id: str
    child_run_id: str
    pin: DelegatePin
    status: BackgroundTaskStatus
    run_status: RunStatus
    output: str | None = None
    output_truncated: bool = False
    error_code: str | None = None


class _BackgroundState:
    def __init__(self, runtime: Runtime[Any], record: _BackgroundTaskRecord) -> None:
        self._state = ScopedStateContext(
            store=runtime.store,
            namespace=record.namespace,
            session_id=record.parent_session_id,
            run_id=record.parent_run_id,
        ).for_session(_BACKGROUND_OWNER)

    async def records(self) -> dict[str, _BackgroundTaskRecord]:
        entries = await self._state.read(
            _BACKGROUND_STATE_KIND,
            schema_version=_BACKGROUND_SCHEMA_VERSION,
        )
        if not entries:
            return {}
        payload = entries[-1].payload
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
            raise BackgroundTaskConflictError("Stored background task state is invalid")
        try:
            return {
                key: _BackgroundTaskRecord.model_validate(value)
                for key, value in payload["tasks"].items()
            }
        except (TypeError, ValueError) as error:
            raise BackgroundTaskConflictError("Stored background task state is invalid") from error

    async def put(self, record: _BackgroundTaskRecord) -> None:
        records = await self.records()
        records[record.operation_key] = record
        entries = await self._state.read(
            _BACKGROUND_STATE_KIND,
            schema_version=_BACKGROUND_SCHEMA_VERSION,
        )
        expected_sequence = entries[-1].sequence if entries else 0
        await self._state.append(
            _BACKGROUND_STATE_KIND,
            {"tasks": {key: value.model_dump(mode="json") for key, value in records.items()}},
            schema_version=_BACKGROUND_SCHEMA_VERSION,
            expected_sequence=expected_sequence,
        )


class BackgroundDelegationService:
    """Start and supervise child Runs without owning an in-process task manager.

    A terminal parent cannot start new background work. Children accepted earlier
    detach and continue until their own terminal Run state is reached or cancelled.
    """

    def __init__(
        self,
        runtime: Runtime[Any],
        *,
        catalog: DelegateCatalog,
        model_for_id: Callable[[str], Model | KnownModelName | str],
        deps_for_request: Callable[[DelegateRequest], tuple[Any, type[Any]]],
        max_task_chars: int = 16_384,
        max_output_bytes: int = 32 * 1024,
        session_lock_timeout: float = 30.0,
    ) -> None:
        if isinstance(max_task_chars, bool) or max_task_chars < 1:
            raise DelegatePolicyError("Background max_task_chars must be positive")
        if isinstance(max_output_bytes, bool) or max_output_bytes < 1:
            raise DelegatePolicyError("Background max_output_bytes must be positive")
        self.runtime = runtime
        self.catalog = catalog
        self.model_for_id = model_for_id
        self.deps_for_request = deps_for_request
        self.max_task_chars = max_task_chars
        self.max_output_bytes = max_output_bytes
        self.session_lock_timeout = session_lock_timeout
        self._steering: dict[str, Callable[[str], Awaitable[None]]] = {}

    async def start(
        self,
        namespace: str,
        parent_session_id: str,
        parent_run_id: str,
        *,
        parent_spec: AgentSpec | Mapping[str, Any],
        alias: str,
        task: str,
        operation_key: str,
        depth: int = 0,
        child_count: int = 0,
    ) -> BackgroundTaskHandle:
        """Persist one accepted child Run before returning its durable handle."""
        _validate_text(task, self.max_task_chars, "Background task")
        _validate_text(operation_key, 255, "Background operation key")
        validated_parent_spec = (
            parent_spec
            if isinstance(parent_spec, AgentSpec)
            else AgentSpec.model_validate(parent_spec)
        )
        selected = self.catalog.resolve_for_invocation(namespace, validated_parent_spec, alias)
        selected.validate_invocation(depth=depth, child_count=child_count)
        parent_session = await self.runtime.load_session(namespace, parent_session_id)
        ensure_session_pin(
            parent_session,
            ExecutionPin.from_spec(validated_parent_spec, model_id=parent_session.pin.model_id),
        )
        parent_run = await self.runtime.get_run(namespace, parent_run_id)
        if parent_run.session_id != parent_session_id:
            raise BackgroundTaskConflictError("Parent Run does not belong to the parent Session")
        if parent_run.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }:
            raise BackgroundTaskConflictError(
                "Background work cannot start from a terminal parent Run"
            )
        template = _BackgroundTaskRecord(
            operation_key=operation_key,
            namespace=namespace,
            parent_session_id=parent_session_id,
            parent_run_id=parent_run_id,
            child_session_id=parent_session_id,
            child_run_id=parent_run_id,
            task=task,
            pin=selected.pin,
            depth=depth,
            child_count=child_count,
        )
        state = _BackgroundState(self.runtime, template)
        async with self.runtime.store.lock_session(
            namespace,
            parent_session_id,
            timeout=self.session_lock_timeout,
        ):
            records = await state.records()
            existing = records.get(operation_key)
            if existing is not None:
                if (
                    existing.task != task
                    or existing.pin != selected.pin
                    or existing.depth != depth
                    or existing.child_count != child_count
                ):
                    raise BackgroundTaskConflictError(
                        "Background operation key has different task, pin, or limits"
                    )
                return await self._view(existing)
            child_session = await self.runtime.create_session(
                namespace,
                spec=selected.spec,
                model_id=selected.model_id,
            )
            reservation = await self.runtime.reserve_run(
                namespace,
                child_session.id,
                task,
                spec=selected.spec,
                model_id=selected.model_id,
                execution_pin=ExecutionPin.from_spec(
                    selected.spec,
                    model_id=selected.model_id,
                ),
                parent_run_id=parent_run_id,
                relation=RunRelation.CHILD,
                idempotency_key=f"background:{operation_key}",
                initiator="delegate_background",
            )
            record = template.model_copy(
                update={
                    "child_session_id": child_session.id,
                    "child_run_id": reservation.run.id,
                }
            )
            await state.put(record)
            return await self._view(record)

    async def inspect(self, handle: BackgroundTaskHandle) -> BackgroundTaskHandle:
        """Reload current state from the parent Session and child Run."""
        record = await self._record(handle)
        return await self._view(record)

    async def list_tasks(
        self, namespace: str, parent_session_id: str
    ) -> list[BackgroundTaskHandle]:
        """List durable child handles for one parent Session."""
        session = await self.runtime.load_session(namespace, parent_session_id)
        template = _BackgroundTaskRecord(
            operation_key="list",
            namespace=namespace,
            parent_session_id=parent_session_id,
            parent_run_id=session.head_run_id or session.id,
            child_session_id=session.id,
            child_run_id=session.head_run_id or session.id,
            task="list",
            pin=DelegatePin(alias="list", version="list", digest="list", model_id="list"),
            depth=0,
            child_count=0,
        )
        records = await _BackgroundState(self.runtime, template).records()
        return [await self._view(record) for record in records.values()]

    async def wait(
        self,
        handle: BackgroundTaskHandle,
        *,
        timeout: float = 30.0,
        poll_interval: float = 0.05,
    ) -> BackgroundTaskHandle:
        """Poll durable Run state; active workers remain external to this service."""
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("Background wait limits must be positive")
        async with asyncio.timeout(timeout):
            while True:
                current = await self.inspect(handle)
                if current.run_status in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.INTERRUPTED,
                }:
                    return current
                await asyncio.sleep(poll_interval)

    async def run_worker(self, handle: BackgroundTaskHandle) -> BackgroundTaskHandle:
        """Claim and execute one child Run; worker loss is recovered by lease expiry."""
        record = await self._record(handle)
        run = await self.runtime.get_run(record.namespace, record.child_run_id)
        if run.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }:
            return await self._view(record)
        selected = self.catalog.resolve_pin(record.namespace, record.pin)
        session = await self.runtime.load_session(record.namespace, record.child_session_id)
        ensure_session_pin(
            session, ExecutionPin.from_spec(selected.spec, model_id=selected.model_id)
        )
        claimed = await self.runtime.store.claim_run(
            record.namespace,
            record.child_run_id,
            owner_id=str(uuid4()),
            lease_seconds=self.runtime.lease_seconds,
        )
        deps, deps_type = self.deps_for_request(
            DelegateRequest(
                namespace=record.namespace,
                parent_session_id=record.parent_session_id,
                parent_run_id=record.parent_run_id,
                root_run_id=record.parent_run_id,
                delegate=selected,
                task=record.task,
                depth=record.depth,
                child_count=record.child_count,
                idempotency_key=f"background:{record.operation_key}",
            )
        )
        await self.runtime.execute_claimed(
            record.namespace,
            claimed,
            prompt=record.task,
            spec=selected.spec,
            model=self.model_for_id(selected.model_id),
            model_id=selected.model_id,
            execution_pin=session.pin,
            deps=deps,
            deps_type=deps_type,
            cancel_on_task_cancel=False,
        )
        return await self._view(record)

    async def cancel(self, handle: BackgroundTaskHandle) -> BackgroundTaskHandle:
        """Persist cancellation through the canonical child Run."""
        record = await self._record(handle)
        await self.runtime.cancel_run(
            record.namespace,
            record.child_run_id,
            error_code="background_cancelled",
            error_detail="Background child cancelled",
        )
        return await self._view(record)

    async def reconcile(self) -> int:
        """Mark expired workers interrupted; no worker registry is required."""
        return await self.runtime.store.interrupt_expired_runs()

    @asynccontextmanager
    async def steering_sink(
        self,
        handle: BackgroundTaskHandle,
        sink: Callable[[str], Awaitable[None]],
    ):
        """Register an ephemeral native enqueue sink for one live worker."""
        self._steering[handle.child_run_id] = sink
        try:
            yield
        finally:
            self._steering.pop(handle.child_run_id, None)

    async def steer(self, handle: BackgroundTaskHandle, prompt: str) -> None:
        """Steer only an active worker through its native enqueue sink."""
        _validate_text(prompt, self.max_task_chars, "Background steering prompt")
        sink = self._steering.get(handle.child_run_id)
        if sink is None:
            raise BackgroundTaskNotActiveError("Background child has no active worker")
        await sink(prompt)

    async def _record(self, handle: BackgroundTaskHandle) -> _BackgroundTaskRecord:
        template = _BackgroundTaskRecord(
            operation_key=handle.operation_key,
            namespace=handle.namespace,
            parent_session_id=handle.parent_session_id,
            parent_run_id=handle.parent_run_id,
            child_session_id=handle.child_session_id,
            child_run_id=handle.child_run_id,
            task="handle",
            pin=handle.pin,
            depth=0,
            child_count=0,
        )
        records = await _BackgroundState(self.runtime, template).records()
        record = records.get(handle.operation_key)
        if record is None or record.child_run_id != handle.child_run_id:
            raise BackgroundTaskNotFoundError("Background task handle is not available")
        return record

    async def _view(self, record: _BackgroundTaskRecord) -> BackgroundTaskHandle:
        run = await self.runtime.get_run(record.namespace, record.child_run_id)
        output, truncated = _bounded_output(run.output, self.max_output_bytes)
        return BackgroundTaskHandle(
            namespace=record.namespace,
            operation_key=record.operation_key,
            parent_session_id=record.parent_session_id,
            parent_run_id=record.parent_run_id,
            child_session_id=record.child_session_id,
            child_run_id=record.child_run_id,
            pin=record.pin,
            status=BackgroundTaskStatus(run.status.value),
            run_status=run.status,
            output=output,
            output_truncated=truncated,
            error_code=run.error_code,
        )


def _validate_text(value: str, limit: int, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DelegatePolicyError(f"{label} must not be empty")
    if len(value) > limit:
        raise DelegatePolicyError(f"{label} exceeds the host limit")


def _bounded_output(value: Any, limit: int) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    encoded = text.encode()
    if len(encoded) <= limit:
        return text, False
    return encoded[:limit].decode(errors="ignore"), True


__all__ = [
    "BackgroundDelegationService",
    "BackgroundTaskConflictError",
    "BackgroundTaskHandle",
    "BackgroundTaskNotActiveError",
    "BackgroundTaskNotFoundError",
    "BackgroundTaskStatus",
]
