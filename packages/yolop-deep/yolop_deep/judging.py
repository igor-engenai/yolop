"""Host-controlled judging and acceptance of experimental fork candidates."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import AgentSpec
from pydantic_ai.models import KnownModelName, Model
from yolop_delegation import bounded_idempotency_key
from yolop_runtime import (
    RunRelation,
    RunStatus,
    Runtime,
    RuntimeSessionSnapshot,
    ScopedStateContext,
)

from .forks import ForkCandidateHandle, ForkCandidateService

_JUDGING_OWNER = "yolop.deep.fork_judging"
_JUDGING_STATE_KIND = "judgments"
_JUDGING_SCHEMA_VERSION = 1


class CandidateJudgeError(ValueError):
    """Candidate evidence or evaluator policy is invalid."""

    code = "fork_candidate_judge_error"


class CandidateAcceptanceError(ValueError):
    """A candidate cannot be accepted under the current source revision."""

    code = "fork_candidate_acceptance_error"


class CandidateVerdict(BaseModel):
    """Bounded evaluator output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: Literal["accept", "reject"]
    reason: str = Field(default="", max_length=2048)


class CandidateJudgment(BaseModel):
    """Durable candidate verdict and evaluator Run identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_key: str
    verdict: Literal["accept", "reject"]
    reason: str = Field(max_length=2048)
    evaluator_session_id: str
    evaluator_run_id: str
    source_revision: str


class _JudgmentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    judgments: dict[str, CandidateJudgment] = Field(default_factory=dict)
    accepted_candidate_key: str | None = None


class _JudgmentState:
    def __init__(self, runtime: Runtime[Any], namespace: str, session_id: str, run_id: str) -> None:
        self._state = ScopedStateContext(
            store=runtime.store,
            namespace=namespace,
            session_id=session_id,
            run_id=run_id,
        ).for_session(_JUDGING_OWNER)

    async def read(self) -> _JudgmentSnapshot:
        entries = await self._state.read(
            _JUDGING_STATE_KIND, schema_version=_JUDGING_SCHEMA_VERSION
        )
        if not entries:
            return _JudgmentSnapshot()
        try:
            return _JudgmentSnapshot.model_validate(entries[-1].payload)
        except (TypeError, ValueError) as error:
            raise CandidateJudgeError("Stored candidate judgment state is invalid") from error

    async def write(self, snapshot: _JudgmentSnapshot) -> None:
        entries = await self._state.read(
            _JUDGING_STATE_KIND, schema_version=_JUDGING_SCHEMA_VERSION
        )
        expected_sequence = entries[-1].sequence if entries else 0
        await self._state.append(
            _JUDGING_STATE_KIND,
            snapshot.model_dump(mode="json"),
            schema_version=_JUDGING_SCHEMA_VERSION,
            expected_sequence=expected_sequence,
        )


class CandidateJudgeService:
    """Evaluate candidates and atomically project one accepted branch into a Session."""

    def __init__(
        self,
        runtime: Runtime[Any],
        *,
        candidates: ForkCandidateService,
        cleanup_candidate: Callable[[ForkCandidateHandle], Awaitable[None]] | None = None,
        max_evidence_bytes: int = 64 * 1024,
    ) -> None:
        if isinstance(max_evidence_bytes, bool) or max_evidence_bytes < 1:
            raise CandidateJudgeError("Candidate evidence limit must be positive")
        self.runtime = runtime
        self.candidates = candidates
        self.cleanup_candidate = cleanup_candidate
        self.max_evidence_bytes = max_evidence_bytes

    async def judge(
        self,
        namespace: str,
        source_session_id: str,
        source_run_id: str,
        *,
        candidate_keys: Sequence[str],
        evaluator_spec: AgentSpec | Mapping[str, Any],
        evaluator_model: Model | KnownModelName | str,
        evaluator_model_id: str,
        deps: Any,
        deps_type: type[Any],
    ) -> tuple[CandidateJudgment, ...]:
        keys = tuple(dict.fromkeys(candidate_keys))
        if not keys:
            raise CandidateJudgeError("Candidate judge requires at least one candidate")
        handles = {
            handle.candidate_key: handle
            for handle in await self.candidates.list_candidates(
                namespace, source_session_id, source_run_id
            )
        }
        selected: list[ForkCandidateHandle] = []
        for key in keys:
            handle = handles.get(key)
            if handle is None:
                raise CandidateJudgeError(f"Candidate {key!r} does not exist")
            if handle.run_status is not RunStatus.COMPLETED:
                raise CandidateJudgeError(f"Candidate {key!r} is not completed")
            selected.append(handle)
        source = await self.runtime.load_session(namespace, source_session_id)
        state = _JudgmentState(self.runtime, namespace, source_session_id, source_run_id)
        existing = await state.read()
        if all(key in existing.judgments for key in keys):
            return tuple(existing.judgments[key] for key in keys)
        evidence = _bounded_evidence(selected, self.max_evidence_bytes)
        validated_evaluator = (
            evaluator_spec
            if isinstance(evaluator_spec, AgentSpec)
            else AgentSpec.model_validate(evaluator_spec)
        )
        evaluator_session = await self.runtime.create_session(
            namespace,
            spec=validated_evaluator,
            model_id=evaluator_model_id,
        )
        completion = await self.runtime.run(
            namespace,
            evaluator_session.id,
            evidence,
            spec=validated_evaluator,
            model=evaluator_model,
            model_id=evaluator_model_id,
            deps=deps,
            deps_type=deps_type,
            output_type=CandidateVerdict,
            parent_run_id=source_run_id,
            relation=RunRelation.CHILD,
            idempotency_key=bounded_idempotency_key("judge", source_run_id, *keys),
            initiator="fork_judge",
        )
        verdict = CandidateVerdict.model_validate(completion.run.output)
        judgments = tuple(
            CandidateJudgment(
                candidate_key=key,
                verdict=verdict.verdict,
                reason=verdict.reason,
                evaluator_session_id=evaluator_session.id,
                evaluator_run_id=completion.run.id,
                source_revision=source.revision,
            )
            for key in keys
        )
        async with self.runtime.store.lock_session(namespace, source_session_id, timeout=30):
            current = await state.read()
            merged = {**current.judgments, **{item.candidate_key: item for item in judgments}}
            await state.write(current.model_copy(update={"judgments": merged}))
        if verdict.verdict == "reject" and self.cleanup_candidate is not None:
            for handle in selected:
                await self.cleanup_candidate(handle)
        return judgments

    async def accept(
        self,
        namespace: str,
        source_session_id: str,
        source_run_id: str,
        candidate_key: str,
        *,
        expected_revision: str,
    ) -> RuntimeSessionSnapshot:
        state = _JudgmentState(self.runtime, namespace, source_session_id, source_run_id)
        async with self.runtime.store.lock_session(namespace, source_session_id, timeout=30):
            snapshot = await state.read()
            judgment = snapshot.judgments.get(candidate_key)
            if judgment is None or judgment.verdict != "accept":
                raise CandidateAcceptanceError("Candidate has no positive judgment")
            if snapshot.accepted_candidate_key not in {None, candidate_key}:
                raise CandidateAcceptanceError("Another candidate has already been accepted")
            source = await self.runtime.load_session(namespace, source_session_id)
            if source.revision != expected_revision:
                raise CandidateAcceptanceError("Source Session changed after judging")
            handles = {
                handle.candidate_key: handle
                for handle in await self.candidates.list_candidates(
                    namespace, source_session_id, source_run_id
                )
            }
            handle = handles.get(candidate_key)
            if handle is None or handle.run_status is not RunStatus.COMPLETED:
                raise CandidateAcceptanceError("Candidate is not available for acceptance")
            candidate_run = await self.runtime.get_run(namespace, handle.candidate_run_id)
            candidate_session = await self.runtime.load_session(
                namespace, handle.candidate_session_id
            )
            if (
                candidate_session.pin != source.pin
                or candidate_run.session_id != handle.candidate_session_id
            ):
                raise CandidateAcceptanceError("Candidate pin or Session identity does not match")
            accepted = await self.runtime.store.replace_session(
                namespace,
                source_session_id,
                expected_revision=expected_revision,
                messages=candidate_run.active_messages,
            )
            await state.write(snapshot.model_copy(update={"accepted_candidate_key": candidate_key}))
            return accepted


def _bounded_evidence(handles: Sequence[ForkCandidateHandle], limit: int) -> str:
    records: list[dict[str, Any]] = []
    size = 0
    for handle in handles:
        item = {
            "candidate_key": handle.candidate_key,
            "candidate_run_id": handle.candidate_run_id,
            "output": handle.output or "",
        }
        encoded = json.dumps(item, ensure_ascii=False).encode()
        if size + len(encoded) > limit:
            break
        records.append(item)
        size += len(encoded)
    if not records:
        raise CandidateJudgeError("Candidate evidence exceeds the host limit")
    return json.dumps({"candidates": records}, ensure_ascii=False)


__all__ = [
    "CandidateAcceptanceError",
    "CandidateJudgeError",
    "CandidateJudgeService",
    "CandidateJudgment",
    "CandidateVerdict",
]
