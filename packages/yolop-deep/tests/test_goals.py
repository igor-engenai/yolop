from __future__ import annotations

from pathlib import Path

from pydantic_ai import AgentSpec
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
