"""Experimental isolated fork candidates above the durable Runtime."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import AgentSpec
from pydantic_ai.models import KnownModelName, Model
from yolop_delegation import bounded_idempotency_key
from yolop_runtime import (
    ExecutionPin,
    RunRelation,
    RunStatus,
    Runtime,
    ScopedStateContext,
    ensure_session_pin,
)

_FORK_OWNER = "yolop.deep.forks"
_FORK_STATE_KIND = "candidates"
_FORK_SCHEMA_VERSION = 1


class CandidateLimitError(ValueError):
    """The host candidate limit has been reached."""

    code = "fork_candidate_limit"


class CandidateConflictError(ValueError):
    """A candidate key has incompatible durable input."""

    code = "fork_candidate_conflict"


class CandidateNotFoundError(ValueError):
    """A fork candidate is not present in the source Session state."""

    code = "fork_candidate_not_found"


class ForkCandidateStatus(StrEnum):
    """Tool-facing candidate status values."""

    ACCEPTED = RunStatus.ACCEPTED.value
    RUNNING = RunStatus.RUNNING.value
    COMPLETED = RunStatus.COMPLETED.value
    FAILED = RunStatus.FAILED.value
    INTERRUPTED = RunStatus.INTERRUPTED.value


class _CandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_key: str = Field(min_length=1, max_length=255)
    namespace: str
    source_session_id: str
    source_run_id: str
    candidate_session_id: str
    candidate_run_id: str
    prompt: str = Field(min_length=1, max_length=16_384)
    pin: ExecutionPin


@dataclass(frozen=True)
class ForkCandidateHandle:
    """Durable candidate identity and bounded current projection."""

    namespace: str
    candidate_key: str
    source_session_id: str
    source_run_id: str
    candidate_session_id: str
    candidate_run_id: str
    pin: ExecutionPin
    status: ForkCandidateStatus
    run_status: RunStatus
    output: str | None = None
    output_truncated: bool = False
    error_code: str | None = None


class _CandidateState:
    def __init__(self, runtime: Runtime[Any], record: _CandidateRecord) -> None:
        self._state = ScopedStateContext(
            store=runtime.store,
            namespace=record.namespace,
            session_id=record.source_session_id,
            run_id=record.source_run_id,
        ).for_session(_FORK_OWNER)

    async def records(self) -> dict[str, _CandidateRecord]:
        entries = await self._state.read(_FORK_STATE_KIND, schema_version=_FORK_SCHEMA_VERSION)
        if not entries:
            return {}
        payload = entries[-1].payload
        if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), dict):
            raise CandidateConflictError("Stored fork candidate state is invalid")
        try:
            return {
                key: _CandidateRecord.model_validate(value)
                for key, value in payload["candidates"].items()
            }
        except (TypeError, ValueError) as error:
            raise CandidateConflictError("Stored fork candidate state is invalid") from error

    async def write(self, records: dict[str, _CandidateRecord]) -> None:
        entries = await self._state.read(_FORK_STATE_KIND, schema_version=_FORK_SCHEMA_VERSION)
        expected_sequence = entries[-1].sequence if entries else 0
        await self._state.append(
            _FORK_STATE_KIND,
            {"candidates": {key: value.model_dump(mode="json") for key, value in records.items()}},
            schema_version=_FORK_SCHEMA_VERSION,
            expected_sequence=expected_sequence,
        )


class ForkCandidateService:
    """Create and supervise isolated experimental candidate Runs."""

    EXPERIMENTAL_WARNING = "Fork candidates are experimental and do not mutate the source Session."

    def __init__(
        self,
        runtime: Runtime[Any],
        *,
        model_for_id: Callable[[str], Model | KnownModelName | str],
        deps_for_candidate: Callable[[_CandidateRecord], tuple[Any, type[Any]]],
        spec_for_pin: Callable[[ExecutionPin], AgentSpec],
        max_candidates: int = 4,
        max_prompt_chars: int = 16_384,
        max_output_bytes: int = 32 * 1024,
    ) -> None:
        if isinstance(max_candidates, bool) or max_candidates < 1:
            raise CandidateLimitError("Fork max_candidates must be positive")
        if isinstance(max_prompt_chars, bool) or max_prompt_chars < 1:
            raise CandidateLimitError("Fork max_prompt_chars must be positive")
        if isinstance(max_output_bytes, bool) or max_output_bytes < 1:
            raise CandidateLimitError("Fork max_output_bytes must be positive")
        self.runtime = runtime
        self.model_for_id = model_for_id
        self.deps_for_candidate = deps_for_candidate
        self.spec_for_pin = spec_for_pin
        self.max_candidates = max_candidates
        self.max_prompt_chars = max_prompt_chars
        self.max_output_bytes = max_output_bytes

    async def start(
        self,
        namespace: str,
        source_session_id: str,
        source_run_id: str,
        *,
        spec: AgentSpec | Mapping[str, Any],
        model_id: str,
        prompt: str,
        candidate_key: str,
    ) -> ForkCandidateHandle:
        if not prompt.strip() or len(prompt) > self.max_prompt_chars:
            raise CandidateLimitError("Fork candidate prompt is empty or exceeds the host limit")
        if not candidate_key.strip() or len(candidate_key) > 255:
            raise CandidateLimitError("Fork candidate key is empty or exceeds the host limit")
        validated_spec = spec if isinstance(spec, AgentSpec) else AgentSpec.model_validate(spec)
        source_session = await self.runtime.load_session(namespace, source_session_id)
        ensure_session_pin(
            source_session, ExecutionPin.from_spec(validated_spec, model_id=model_id)
        )
        source_run = await self.runtime.get_run(namespace, source_run_id)
        if source_run.session_id != source_session_id:
            raise CandidateConflictError("Source Run does not belong to the source Session")
        if source_run.status not in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }:
            raise CandidateConflictError("Fork source must be a terminal Run checkpoint")
        template = _CandidateRecord(
            candidate_key=candidate_key,
            namespace=namespace,
            source_session_id=source_session_id,
            source_run_id=source_run_id,
            candidate_session_id=source_session_id,
            candidate_run_id=source_run_id,
            prompt=prompt,
            pin=ExecutionPin.from_spec(validated_spec, model_id=model_id),
        )
        state = _CandidateState(self.runtime, template)
        records = await state.records()
        existing = records.get(candidate_key)
        if existing is not None:
            if existing.prompt != prompt or existing.pin != template.pin:
                raise CandidateConflictError("Fork candidate key has different input")
            return await self._view(existing)
        if len(records) >= self.max_candidates:
            raise CandidateLimitError("Fork candidate limit has been reached")
        candidate_session = await self.runtime.fork_session(
            namespace,
            source_session_id,
            source_run_id,
            expected_revision=source_session.revision,
        )
        reservation = await self.runtime.reserve_run(
            namespace,
            candidate_session.id,
            prompt,
            spec=validated_spec,
            model_id=model_id,
            execution_pin=template.pin,
            parent_run_id=source_run_id,
            relation=RunRelation.CHILD,
            idempotency_key=bounded_idempotency_key("fork", candidate_key),
            initiator="fork_candidate",
        )
        record = template.model_copy(
            update={
                "candidate_session_id": candidate_session.id,
                "candidate_run_id": reservation.run.id,
            }
        )
        await state.write({**records, candidate_key: record})
        return await self._view(record)

    async def inspect(self, handle: ForkCandidateHandle) -> ForkCandidateHandle:
        record = await self._record(handle)
        return await self._view(record)

    async def list_candidates(
        self, namespace: str, source_session_id: str, source_run_id: str
    ) -> list[ForkCandidateHandle]:
        template = _CandidateRecord(
            candidate_key="list",
            namespace=namespace,
            source_session_id=source_session_id,
            source_run_id=source_run_id,
            candidate_session_id=source_session_id,
            candidate_run_id=source_run_id,
            prompt="list",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="list"),
        )
        records = await _CandidateState(self.runtime, template).records()
        return [await self._view(record) for record in records.values()]

    async def run(self, handle: ForkCandidateHandle) -> ForkCandidateHandle:
        record = await self._record(handle)
        run = await self.runtime.get_run(record.namespace, record.candidate_run_id)
        if run.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }:
            return await self._view(record)
        session = await self.runtime.load_session(record.namespace, record.candidate_session_id)
        candidate_spec = self.spec_for_pin(record.pin)
        ensure_session_pin(session, record.pin)
        if ExecutionPin.from_spec(candidate_spec, model_id=record.pin.model_id) != record.pin:
            raise CandidateConflictError("Candidate AgentSpec no longer matches its pin")
        claimed = await self.runtime.store.claim_run(
            record.namespace,
            record.candidate_run_id,
            owner_id=str(uuid4()),
            lease_seconds=self.runtime.lease_seconds,
        )
        deps, deps_type = self.deps_for_candidate(record)
        await self.runtime.execute_claimed(
            record.namespace,
            claimed,
            prompt=record.prompt,
            spec=candidate_spec,
            model=self.model_for_id(record.pin.model_id),
            model_id=record.pin.model_id,
            execution_pin=session.pin,
            deps=deps,
            deps_type=deps_type,
            cancel_on_task_cancel=False,
        )
        return await self._view(record)

    async def _record(self, handle: ForkCandidateHandle) -> _CandidateRecord:
        template = _CandidateRecord(
            candidate_key=handle.candidate_key,
            namespace=handle.namespace,
            source_session_id=handle.source_session_id,
            source_run_id=handle.source_run_id,
            candidate_session_id=handle.candidate_session_id,
            candidate_run_id=handle.candidate_run_id,
            prompt="handle",
            pin=handle.pin,
        )
        record = (await _CandidateState(self.runtime, template).records()).get(handle.candidate_key)
        if record is None or record.candidate_run_id != handle.candidate_run_id:
            raise CandidateNotFoundError("Fork candidate handle is not available")
        return record

    async def _view(self, record: _CandidateRecord) -> ForkCandidateHandle:
        run = await self.runtime.get_run(record.namespace, record.candidate_run_id)
        output, truncated = _bounded_output(run.output, self.max_output_bytes)
        return ForkCandidateHandle(
            namespace=record.namespace,
            candidate_key=record.candidate_key,
            source_session_id=record.source_session_id,
            source_run_id=record.source_run_id,
            candidate_session_id=record.candidate_session_id,
            candidate_run_id=record.candidate_run_id,
            pin=record.pin,
            status=ForkCandidateStatus(run.status.value),
            run_status=run.status,
            output=output,
            output_truncated=truncated,
            error_code=run.error_code,
        )


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
    "CandidateConflictError",
    "CandidateLimitError",
    "CandidateNotFoundError",
    "ForkCandidateHandle",
    "ForkCandidateService",
    "ForkCandidateStatus",
]
