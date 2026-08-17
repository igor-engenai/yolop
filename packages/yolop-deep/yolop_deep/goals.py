"""Durable bounded goal continuation above the generic YoloP Runtime."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import KnownModelName, Model
from yolop_runtime import (
    ExecutionPin,
    RunBudgetExceededError,
    RunCompletion,
    RunRelation,
    Runtime,
    RuntimeBudget,
    RuntimeDeadlineExceededError,
    RuntimeRunSnapshot,
    ScopedStateContext,
)

_GOAL_OWNER = "yolop.deep.goals"
_GOAL_STATE_KIND = "goals"
_GOAL_SCHEMA_VERSION = 1


class GoalStatus(StrEnum):
    """Durable lifecycle of one goal policy record."""

    ACTIVE = "active"
    MET = "met"
    IMPOSSIBLE = "impossible"
    EXHAUSTED = "exhausted"
    STOPPED = "stopped"
    FAILED = "failed"


class GoalVerdict(BaseModel):
    """Typed evaluator result based on transcript evidence."""

    verdict: str
    reason: str = ""

    def normalized(self) -> str:
        value = self.verdict.strip().lower()
        if value not in {"met", "impossible", "unmet"}:
            raise ValueError(f"Unsupported goal verdict: {self.verdict!r}")
        return value


class GoalRecord(BaseModel):
    """Persisted goal policy state, not an execution entity."""

    goal_id: str
    namespace: str
    session_id: str
    condition: str
    status: GoalStatus
    evaluator_spec: dict[str, Any]
    evaluator_model_id: str
    reason: str = ""
    turns: int = 0
    max_turns: int = Field(gt=0)
    root_run_id: str | None = None
    last_run_id: str | None = None
    evaluator_run_id: str | None = None
    needs_evaluation: bool = False
    deadline: str | None = None
    budget: dict[str, Any] = Field(default_factory=dict)


class GoalStore:
    """Append-only RuntimeStore reducer for Session-scoped goal records."""

    def __init__(self, state: ScopedStateContext) -> None:
        self._state = state.for_session(_GOAL_OWNER)

    async def get(self, goal_id: str) -> GoalRecord | None:
        records = await self._read()
        return records.get(goal_id)

    async def put(self, record: GoalRecord) -> GoalRecord:
        records = await self._read()
        records[record.goal_id] = record.model_copy(deep=True)
        entries = await self._state.read(
            _GOAL_STATE_KIND,
            schema_version=_GOAL_SCHEMA_VERSION,
        )
        expected_sequence = entries[-1].sequence if entries else 0
        await self._state.append(
            _GOAL_STATE_KIND,
            {"goals": {key: value.model_dump(mode="json") for key, value in records.items()}},
            schema_version=_GOAL_SCHEMA_VERSION,
            expected_sequence=expected_sequence,
        )
        return record.model_copy(deep=True)

    async def _read(self) -> dict[str, GoalRecord]:
        entries = await self._state.read(
            _GOAL_STATE_KIND,
            schema_version=_GOAL_SCHEMA_VERSION,
        )
        if not entries:
            return {}
        payload = entries[-1].payload
        if not isinstance(payload, dict) or not isinstance(payload.get("goals"), dict):
            raise ValueError("Stored goal state has an invalid snapshot")
        try:
            return {
                goal_id: GoalRecord.model_validate(value)
                for goal_id, value in payload["goals"].items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("Stored goal state has an invalid record") from error


class GoalRunner:
    """Run bounded goals as ordinary Runtime related Runs."""

    def __init__(self, runtime: Runtime[Any]) -> None:
        self.runtime = runtime

    async def start(
        self,
        namespace: str,
        session_id: str,
        *,
        goal: str,
        spec: AgentSpec | dict[str, Any],
        model: Model | KnownModelName | str,
        model_id: str,
        evaluator_spec: AgentSpec | dict[str, Any],
        evaluator_model: Model | KnownModelName | str,
        evaluator_model_id: str,
        deps: Any,
        deps_type: type[Any],
        budget: RuntimeBudget | None = None,
        max_turns: int = 3,
    ) -> GoalRecord:
        """Start a goal, evaluate its first transcript, and schedule continuation."""
        if not goal.strip():
            raise ValueError("Goal condition must not be empty")
        if max_turns < 1:
            raise ValueError("Goal max_turns must be positive")
        validated_evaluator = _validated_evaluator_spec(evaluator_spec)
        evaluator_pin = ExecutionPin.from_spec(
            validated_evaluator,
            model_id=evaluator_model_id,
        )
        record = GoalRecord(
            goal_id=str(uuid4()),
            namespace=namespace,
            session_id=session_id,
            condition=goal,
            status=GoalStatus.ACTIVE,
            evaluator_spec=validated_evaluator.model_dump(mode="json"),
            evaluator_model_id=evaluator_pin.model_id,
            max_turns=max_turns,
            deadline=_deadline_text(budget),
            budget=_budget_payload(budget),
        )
        store = await self._store(namespace, session_id, record.goal_id)
        await store.put(record)
        try:
            completion = await self.runtime.run(
                namespace,
                session_id,
                goal,
                spec=spec,
                model=model,
                model_id=model_id,
                root_budget=budget,
                deps=deps,
                deps_type=deps_type,
                idempotency_key=f"goal:{record.goal_id}:turn:1",
                initiator="goal",
            )
        except Exception as error:
            record.status = GoalStatus.FAILED
            record.reason = f"goal run failed: {type(error).__name__}"
            await store.put(record)
            raise
        record.root_run_id = completion.run.root_run_id or completion.run.id
        record.last_run_id = completion.run.id
        record.turns = 1
        record.needs_evaluation = True
        await store.put(record)
        return await self._advance(
            store,
            record,
            completion,
            spec=spec,
            model=model,
            model_id=model_id,
            evaluator_model=evaluator_model,
            deps=deps,
            deps_type=deps_type,
        )

    async def get(self, namespace: str, session_id: str, goal_id: str) -> GoalRecord | None:
        store = await self._store(namespace, session_id, goal_id)
        return await store.get(goal_id)

    async def stop(
        self, namespace: str, session_id: str, goal_id: str, *, reason: str = "stopped"
    ) -> GoalRecord:
        store = await self._store(namespace, session_id, goal_id)
        record = await store.get(goal_id)
        if record is None:
            raise ValueError(f"Goal {goal_id!r} does not exist")
        if record.status is GoalStatus.ACTIVE:
            record.status = GoalStatus.STOPPED
            record.reason = reason
            record.needs_evaluation = False
            await store.put(record)
        return record

    async def resume(
        self,
        namespace: str,
        session_id: str,
        goal_id: str,
        *,
        spec: AgentSpec | dict[str, Any],
        model: Model | KnownModelName | str,
        model_id: str,
        evaluator_model: Model | KnownModelName | str,
        deps: Any,
        deps_type: type[Any],
    ) -> GoalRecord:
        """Resume an active goal after restart, idempotently."""
        store = await self._store(namespace, session_id, goal_id)
        record = await store.get(goal_id)
        if record is None:
            raise ValueError(f"Goal {goal_id!r} does not exist")
        if record.status is not GoalStatus.ACTIVE or not record.needs_evaluation:
            return record
        if record.last_run_id is None:
            raise ValueError("Active goal has no last Run")
        run = await self.runtime.get_run(namespace, record.last_run_id)
        session = await self.runtime.load_session(namespace, session_id)
        completion = RunCompletion(session=session, run=run)
        evaluator_spec = AgentSpec.model_validate(record.evaluator_spec)
        return await self._advance(
            store,
            record,
            completion,
            spec=spec,
            model=model,
            model_id=model_id,
            evaluator_model=evaluator_model,
            deps=deps,
            deps_type=deps_type,
            evaluator_spec=evaluator_spec,
        )

    async def _advance(
        self,
        store: GoalStore,
        record: GoalRecord,
        completion: RunCompletion,
        *,
        spec: AgentSpec | dict[str, Any],
        model: Model | KnownModelName | str,
        model_id: str,
        evaluator_model: Model | KnownModelName | str,
        deps: Any,
        deps_type: type[Any],
        evaluator_spec: AgentSpec | None = None,
    ) -> GoalRecord:
        evaluator_spec = evaluator_spec or AgentSpec.model_validate(record.evaluator_spec)
        record.needs_evaluation = False
        evaluator_completion: RunCompletion | None = None
        try:
            evaluator_completion = await self.runtime.run_related(
                record.namespace,
                record.session_id,
                _evaluator_prompt(record.condition, completion.run),
                parent_run_id=completion.run.id,
                relation=RunRelation.CHILD,
                spec=evaluator_spec,
                model=evaluator_model,
                model_id=record.evaluator_model_id,
                execution_pin=ExecutionPin.from_spec(
                    evaluator_spec,
                    model_id=record.evaluator_model_id,
                ),
                deps=deps,
                deps_type=deps_type,
                idempotency_key=f"goal:{record.goal_id}:evaluation:{record.turns}",
                initiator="goal",
            )
            verdict = GoalVerdict.model_validate(evaluator_completion.run.output)
            normalized = verdict.normalized()
        except Exception as error:
            normalized = "unmet"
            reason = f"evaluator failed: {type(error).__name__}"
        else:
            reason = verdict.reason
            record.evaluator_run_id = evaluator_completion.run.id
        if normalized == "met":
            record.status = GoalStatus.MET
            record.reason = reason
            await store.put(record)
            return record
        if normalized == "impossible":
            record.status = GoalStatus.IMPOSSIBLE
            record.reason = reason
            await store.put(record)
            return record
        if record.turns >= record.max_turns:
            record.status = GoalStatus.EXHAUSTED
            record.reason = reason or "maximum goal turns reached"
            await store.put(record)
            return record
        parent_run_id = (
            evaluator_completion.run.id if evaluator_completion is not None else completion.run.id
        )
        try:
            continuation = await self.runtime.run_related(
                record.namespace,
                record.session_id,
                _continuation_prompt(record.condition, reason),
                parent_run_id=parent_run_id,
                relation=RunRelation.CONTINUATION,
                spec=spec,
                model=model,
                model_id=model_id,
                deps=deps,
                deps_type=deps_type,
                idempotency_key=f"goal:{record.goal_id}:turn:{record.turns + 1}",
                initiator="goal",
            )
        except (RunBudgetExceededError, RuntimeDeadlineExceededError) as error:
            record.status = GoalStatus.EXHAUSTED
            record.reason = f"goal bound reached: {type(error).__name__}"
            await store.put(record)
            return record
        record.turns += 1
        record.last_run_id = continuation.run.id
        record.reason = reason
        record.needs_evaluation = True
        await store.put(record)
        return record

    async def _store(self, namespace: str, session_id: str, run_id: str) -> GoalStore:
        return GoalStore(
            ScopedStateContext(
                store=self.runtime.store,
                namespace=namespace,
                session_id=session_id,
                run_id=run_id,
            )
        )


def _validated_evaluator_spec(spec: AgentSpec | dict[str, Any]) -> AgentSpec:
    validated = spec if isinstance(spec, AgentSpec) else AgentSpec.model_validate(spec)
    if validated.capabilities:
        raise ValueError("Goal evaluator AgentSpec must not select tools or capabilities")
    return validated


def _budget_payload(budget: RuntimeBudget | None) -> dict[str, Any]:
    if budget is None:
        return {}
    payload = asdict(budget)
    if isinstance(payload.get("wall_deadline"), datetime):
        payload["wall_deadline"] = payload["wall_deadline"].isoformat()
    return payload


def _deadline_text(budget: RuntimeBudget | None) -> str | None:
    if budget is None or budget.wall_deadline is None:
        return None
    return budget.wall_deadline.isoformat()


def _evaluator_prompt(condition: str, run: RuntimeRunSnapshot) -> str:
    evidence = ModelMessagesTypeAdapter.dump_json(run.full_messages).decode()
    return (
        "Evaluate the goal using only the transcript evidence below. Do not call tools.\n"
        f"Goal: {condition}\nTranscript evidence:\n{evidence}\n"
        "Return verdict met, impossible, or unmet, with a concrete reason."
    )


def _continuation_prompt(condition: str, reason: str) -> str:
    return f"Continue toward this goal: {condition}\nEvaluator reason: {reason}"


__all__ = ["GoalRecord", "GoalRunner", "GoalStatus", "GoalStore", "GoalVerdict"]
