"""Host-controlled judging and acceptance of experimental fork candidates."""

from __future__ import annotations

import asyncio
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
    RuntimeRunSnapshot,
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
    """One bounded evaluator decision for one candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_key: str = Field(min_length=1, max_length=255)
    verdict: Literal["accept", "reject"]
    reason: str = Field(default="", max_length=2048)


class CandidateVerdictSet(BaseModel):
    """Evaluator output containing one decision for every requested candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judgments: tuple[CandidateVerdict, ...] = Field(min_length=1)


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
    accepted_source_revision: str | None = None


class _JudgmentState:
    def __init__(self, runtime: Runtime[Any], namespace: str, session_id: str, run_id: str) -> None:
        self._state = ScopedStateContext(
            store=runtime.store,
            namespace=namespace,
            session_id=session_id,
            run_id=run_id,
        ).for_run(_JUDGING_OWNER)

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
        self._evaluation_locks: dict[str, asyncio.Lock] = {}

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
        evaluation_key = bounded_idempotency_key("judge", source_run_id, *keys)
        lock = self._evaluation_locks.setdefault(evaluation_key, asyncio.Lock())
        async with lock:
            existing = await state.read()
            if all(key in existing.judgments for key in keys):
                return tuple(existing.judgments[key] for key in keys)
            evidence = _bounded_evidence(selected, self.max_evidence_bytes)
            evaluator_run = await _find_evaluator_run(
                self.runtime,
                namespace,
                evaluation_key,
                parent_run_id=source_run_id,
            )
            if evaluator_run is not None:
                if evaluator_run.status is not RunStatus.COMPLETED:
                    raise CandidateJudgeError("The evaluator Run is not completed")
            else:
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
                    output_type=CandidateVerdictSet,
                    parent_run_id=source_run_id,
                    relation=RunRelation.CHILD,
                    idempotency_key=evaluation_key,
                    initiator="fork_judge",
                )
                evaluator_run = completion.run
            verdicts = _verdicts_for_keys(evaluator_run.output, keys)
            judgments = tuple(
                CandidateJudgment(
                    candidate_key=key,
                    verdict=verdicts[key].verdict,
                    reason=verdicts[key].reason,
                    evaluator_session_id=evaluator_run.session_id,
                    evaluator_run_id=evaluator_run.id,
                    source_revision=source.revision,
                )
                for key in keys
            )
            current = await state.read()
            merged = {**current.judgments, **{item.candidate_key: item for item in judgments}}
            await state.write(current.model_copy(update={"judgments": merged}))
        if self.cleanup_candidate is not None:
            for handle, judgment in zip(selected, judgments, strict=True):
                if judgment.verdict == "reject":
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
            if snapshot.accepted_candidate_key == candidate_key:
                if source.messages == candidate_run.active_messages:
                    if source.revision != expected_revision:
                        raise CandidateAcceptanceError("Source Session revision is stale")
                    return source
                if (
                    source.revision != expected_revision
                    or snapshot.accepted_source_revision != source.revision
                ):
                    raise CandidateAcceptanceError("Source Session changed during acceptance")
            else:
                if (
                    source.revision != expected_revision
                    or judgment.source_revision != expected_revision
                ):
                    raise CandidateAcceptanceError("Source Session changed after judging")
                await state.write(
                    snapshot.model_copy(
                        update={
                            "accepted_candidate_key": candidate_key,
                            "accepted_source_revision": source.revision,
                        }
                    )
                )
            return await self.runtime.store.replace_session(
                namespace,
                source_session_id,
                expected_revision=source.revision,
                messages=candidate_run.active_messages,
            )


async def _find_evaluator_run(
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


def _verdicts_for_keys(output: Any, keys: tuple[str, ...]) -> dict[str, CandidateVerdict]:
    try:
        verdict_set = CandidateVerdictSet.model_validate(output)
    except (TypeError, ValueError) as error:
        raise CandidateJudgeError("Evaluator output is not a candidate verdict set") from error
    verdicts = {verdict.candidate_key: verdict for verdict in verdict_set.judgments}
    if set(verdicts) != set(keys) or len(verdicts) != len(verdict_set.judgments):
        raise CandidateJudgeError("Evaluator must return exactly one verdict per candidate")
    return verdicts


def _bounded_evidence(handles: Sequence[ForkCandidateHandle], limit: int) -> str:
    records = [
        {
            "candidate_key": handle.candidate_key,
            "candidate_run_id": handle.candidate_run_id,
            "output": handle.output or "",
        }
        for handle in handles
    ]
    evidence = json.dumps({"candidates": records}, ensure_ascii=False)
    if len(evidence.encode()) > limit:
        raise CandidateJudgeError("Candidate evidence exceeds the host limit")
    return evidence


__all__ = [
    "CandidateAcceptanceError",
    "CandidateJudgeError",
    "CandidateJudgeService",
    "CandidateJudgment",
    "CandidateVerdict",
    "CandidateVerdictSet",
]
