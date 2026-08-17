from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
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


async def test_goals_are_isolated_by_session_and_namespace(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(model="test:evaluator", instructions="Return a verdict.")
    first = await runtime.create_session("tenant-a", spec=work_spec, model_id="test:model")
    second = await runtime.create_session("tenant-a", spec=work_spec, model_id="test:model")
    other = await runtime.create_session("tenant-b", spec=work_spec, model_id="test:model")
    runner = GoalRunner(runtime)

    async def start(namespace: str, session_id: str, text: str):
        return await runner.start(
            namespace,
            session_id,
            goal=text,
            spec=work_spec,
            model=TestModel(custom_output_text="attempted"),
            model_id="test:model",
            evaluator_spec=evaluator_spec,
            evaluator_model=TestModel(custom_output_args={"verdict": "met", "reason": "done"}),
            evaluator_model_id="test:evaluator",
            deps=None,
            deps_type=type(None),
        )

    first_goal = await start("tenant-a", first.id, "first")
    second_goal = await start("tenant-a", second.id, "second")
    other_goal = await start("tenant-b", other.id, "other")

    assert await runner.get("tenant-a", first.id, first_goal.goal_id) is not None
    assert await runner.get("tenant-a", first.id, second_goal.goal_id) is None
    assert await runner.get("tenant-b", other.id, other_goal.goal_id) is not None
    assert await runner.get("tenant-b", other.id, first_goal.goal_id) is None


async def test_process_restart_resumes_active_goal_once(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    first_store = SQLiteRuntimeStore(database)
    first_runtime = Runtime(store=first_store)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(model="test:evaluator", instructions="Return a verdict.")
    session = await first_runtime.create_session("tenant", spec=work_spec, model_id="test:model")
    evaluator_model = TestModel(custom_output_args={"verdict": "unmet", "reason": "continue"})
    first = await GoalRunner(first_runtime).start(
        "tenant",
        session.id,
        goal="Continue after restart.",
        spec=work_spec,
        model=TestModel(custom_output_text="attempted"),
        model_id="test:model",
        evaluator_spec=evaluator_spec,
        evaluator_model=evaluator_model,
        evaluator_model_id="test:evaluator",
        deps=None,
        deps_type=type(None),
        max_turns=3,
    )
    before = await first_runtime.list_runs("tenant", session_id=session.id)

    reopened_runtime = Runtime(store=SQLiteRuntimeStore(database))
    resumed = await GoalRunner(reopened_runtime).resume(
        "tenant",
        session.id,
        first.goal_id,
        spec=work_spec,
        model=TestModel(custom_output_text="resumed"),
        model_id="test:model",
        evaluator_model=evaluator_model,
        deps=None,
        deps_type=type(None),
    )
    after = await reopened_runtime.list_runs("tenant", session_id=session.id)

    assert resumed.turns == 3
    assert resumed.status is GoalStatus.ACTIVE
    assert len(after) == len(before) + 2
    assert len({run.id for run in after}) == len(after)


async def test_stopping_goal_prevents_future_continuation(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(model="test:evaluator", instructions="Return a verdict.")
    session = await runtime.create_session("tenant", spec=work_spec, model_id="test:model")
    evaluator_model = TestModel(custom_output_args={"verdict": "unmet", "reason": "continue later"})
    runner = GoalRunner(runtime)
    record = await runner.start(
        "tenant",
        session.id,
        goal="Stop before resuming.",
        spec=work_spec,
        model=TestModel(custom_output_text="attempted"),
        model_id="test:model",
        evaluator_spec=evaluator_spec,
        evaluator_model=evaluator_model,
        evaluator_model_id="test:evaluator",
        deps=None,
        deps_type=type(None),
        max_turns=3,
    )
    stopped = await runner.stop("tenant", session.id, record.goal_id)
    before = len(await runtime.list_runs("tenant", session_id=session.id))

    resumed = await runner.resume(
        "tenant",
        session.id,
        record.goal_id,
        spec=work_spec,
        model=TestModel(custom_output_text="should not run"),
        model_id="test:model",
        evaluator_model=evaluator_model,
        deps=None,
        deps_type=type(None),
    )

    assert stopped.status is GoalStatus.STOPPED
    assert resumed.status is GoalStatus.STOPPED
    assert len(await runtime.list_runs("tenant", session_id=session.id)) == before


async def test_root_deadline_stops_goal_continuation(tmp_path: Path) -> None:
    current = datetime.now(UTC)
    deadline = current + timedelta(seconds=10)

    class Clock:
        value = current

        def __call__(self) -> datetime:
            return self.value

    clock = Clock()

    async def work(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        clock.value = deadline + timedelta(seconds=1)
        yield "attempted"

    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store, clock=clock)
    work_spec = AgentSpec(model="test:model")
    evaluator_spec = AgentSpec(model="test:evaluator", instructions="Return a verdict.")
    session = await runtime.create_session("tenant", spec=work_spec, model_id="test:model")

    record = await GoalRunner(runtime).start(
        "tenant",
        session.id,
        goal="Meet the deadline.",
        spec=work_spec,
        model=FunctionModel(stream_function=work),
        model_id="test:model",
        evaluator_spec=evaluator_spec,
        evaluator_model=TestModel(custom_output_args={"verdict": "unmet", "reason": "not done"}),
        evaluator_model_id="test:evaluator",
        deps=None,
        deps_type=type(None),
        budget=RuntimeBudget(wall_deadline=deadline, request_limit=10),
        max_turns=3,
    )

    assert record.status is GoalStatus.EXHAUSTED
    assert "RuntimeDeadlineExceededError" in record.reason
    assert len(await runtime.list_runs("tenant", session_id=session.id)) == 1


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
