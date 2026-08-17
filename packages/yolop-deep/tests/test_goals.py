from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from yolop_deep import GoalRunner, GoalStatus
from yolop_runtime import RunRelation, Runtime, RuntimeBudget
from yolop_sqlite_session import SQLiteRuntimeStore


async def test_met_goal_stops_without_a_continuation(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(
        model="test:evaluator",
        output_schema={
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "reason"],
        },
    )
    session = await runtime.create_session("tenant", spec=work_spec, model_id="test:model")

    record = await GoalRunner(runtime).start(
        "tenant",
        session.id,
        goal="Verify the change.",
        spec=work_spec,
        model=TestModel(custom_output_text="verified"),
        model_id="test:model",
        evaluator_spec=evaluator_spec,
        evaluator_model=TestModel(
            custom_output_args={"verdict": "met", "reason": "verification is concrete"}
        ),
        evaluator_model_id="test:evaluator",
        deps=None,
        deps_type=type(None),
        max_turns=3,
    )

    assert record.status is GoalStatus.MET
    assert record.reason == "verification is concrete"
    runs = await runtime.list_runs("tenant", session_id=session.id)
    assert len(runs) == 2


async def test_impossible_goal_stops_with_reason(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(
        model="test:evaluator",
        output_schema={
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "reason"],
        },
    )
    session = await runtime.create_session("tenant", spec=work_spec, model_id="test:model")

    record = await GoalRunner(runtime).start(
        "tenant",
        session.id,
        goal="Reach an unavailable service.",
        spec=work_spec,
        model=TestModel(custom_output_text="attempted"),
        model_id="test:model",
        evaluator_spec=evaluator_spec,
        evaluator_model=TestModel(
            custom_output_args={"verdict": "impossible", "reason": "service is offline"}
        ),
        evaluator_model_id="test:evaluator",
        deps=None,
        deps_type=type(None),
        max_turns=3,
    )

    assert record.status is GoalStatus.IMPOSSIBLE
    assert record.reason == "service is offline"
    assert len(await runtime.list_runs("tenant", session_id=session.id)) == 2


async def test_root_budget_stops_goal_continuation(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(
        model="test:evaluator",
        output_schema={
            "type": "object",
            "properties": {"verdict": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["verdict", "reason"],
        },
    )
    session = await runtime.create_session("tenant", spec=work_spec, model_id="test:model")

    record = await GoalRunner(runtime).start(
        "tenant",
        session.id,
        goal="Stay within the request budget.",
        spec=work_spec,
        model=TestModel(custom_output_text="attempted"),
        model_id="test:model",
        evaluator_spec=evaluator_spec,
        evaluator_model=TestModel(custom_output_args={"verdict": "unmet", "reason": "not done"}),
        evaluator_model_id="test:evaluator",
        deps=None,
        deps_type=type(None),
        budget=RuntimeBudget(request_limit=2, child_run_limit=5, continuation_limit=3),
        max_turns=3,
    )

    assert record.status is GoalStatus.EXHAUSTED
    assert "RunBudgetExceededError" in record.reason
    assert len(await runtime.list_runs("tenant", session_id=session.id)) == 2


async def test_maximum_turns_stops_unmet_goal(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(
        model="test:evaluator",
        output_schema={
            "type": "object",
            "properties": {"verdict": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["verdict", "reason"],
        },
    )
    session = await runtime.create_session("tenant", spec=work_spec, model_id="test:model")

    record = await GoalRunner(runtime).start(
        "tenant",
        session.id,
        goal="Do not stop yet.",
        spec=work_spec,
        model=TestModel(custom_output_text="attempted"),
        model_id="test:model",
        evaluator_spec=evaluator_spec,
        evaluator_model=TestModel(custom_output_args={"verdict": "unmet", "reason": "not done"}),
        evaluator_model_id="test:evaluator",
        deps=None,
        deps_type=type(None),
        max_turns=1,
    )

    assert record.status is GoalStatus.EXHAUSTED
    assert record.reason == "not done"
    assert len(await runtime.list_runs("tenant", session_id=session.id)) == 2


async def test_evaluator_failure_continues_within_the_goal_budget(tmp_path: Path) -> None:
    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("evaluator unavailable")
        yield "unreachable"

    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(model="test:evaluator", instructions="Return a verdict.")
    session = await runtime.create_session("tenant", spec=work_spec, model_id="test:model")

    record = await GoalRunner(runtime).start(
        "tenant",
        session.id,
        goal="Keep trying.",
        spec=work_spec,
        model=TestModel(custom_output_text="attempted"),
        model_id="test:model",
        evaluator_spec=evaluator_spec,
        evaluator_model=FunctionModel(stream_function=fail),
        evaluator_model_id="test:evaluator",
        deps=None,
        deps_type=type(None),
        budget=RuntimeBudget(request_limit=10, child_run_limit=5, continuation_limit=3),
        max_turns=2,
    )

    assert record.status is GoalStatus.ACTIVE
    assert "evaluator failed" in record.reason
    runs = await runtime.list_runs("tenant", session_id=session.id)
    assert len(runs) == 3
    assert runs[1].status.value == "failed"


async def test_unmet_goal_schedules_one_related_continuation(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(
        model="test:evaluator",
        output_schema={
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["met", "impossible", "unmet"]},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "reason"],
        },
    )
    session = await runtime.create_session("tenant", spec=work_spec, model_id="test:model")
    runner = GoalRunner(runtime)

    record = await runner.start(
        "tenant",
        session.id,
        goal="Make the change and verify it.",
        spec=work_spec,
        model=TestModel(custom_output_text="progress"),
        model_id="test:model",
        evaluator_spec=evaluator_spec,
        evaluator_model=TestModel(
            custom_output_args={"verdict": "unmet", "reason": "verification remains"}
        ),
        evaluator_model_id="test:evaluator",
        deps=None,
        deps_type=type(None),
        budget=RuntimeBudget(request_limit=10, child_run_limit=5, continuation_limit=3),
        max_turns=2,
    )

    assert record.status is GoalStatus.ACTIVE
    assert record.turns == 2
    runs = await runtime.list_runs("tenant", session_id=session.id)
    assert len(runs) == 3
    assert [run.relation for run in runs] == [
        RunRelation.ROOT,
        RunRelation.CHILD,
        RunRelation.CONTINUATION,
    ]
    assert all(run.initiator == "goal" for run in runs)
    assert len({run.root_run_id for run in runs}) == 1
