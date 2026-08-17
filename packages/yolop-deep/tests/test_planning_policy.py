from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.planning import PlanningToolset
from yolop_deep import PlanItem, Planning, SessionPlanStore, TaskStatus
from yolop_runtime import ExecutionPin, ExecutionScope, RuntimeDeps, ScopedStateContext
from yolop_sqlite_session import SQLiteRuntimeStore

from yolop import ProviderCatalog


async def test_subtask_dependencies_hide_blocked_work(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime_session = await store.create_session(
        "tenant",
        pin=ExecutionPin(agent_spec_id="spec", model_id="test:model"),
    )
    state = ScopedStateContext(
        store=store,
        namespace="tenant",
        session_id=runtime_session.id,
        run_id=runtime_session.id,
    )
    plan = SessionPlanStore(state)
    await plan.set_items(
        [
            PlanItem(id="first", content="finish first"),
            PlanItem(id="second", content="then second"),
        ]
    )
    deps = RuntimeDeps(
        scope=ExecutionScope(
            namespace="tenant",
            session_id=runtime_session.id,
            run_id=runtime_session.id,
        ),
        state=state,
        event_sink=None,
        follow_up_sink=None,
        host=None,
    )
    context = RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        run_id=runtime_session.id,
    )
    capability = await Planning.from_spec(enable_subtasks=True).for_run(context)
    toolset = cast(PlanningToolset[Any], capability.get_toolset())

    message = await toolset.set_dependency(context, "second", "first")
    assert "blocked" in message
    assert "second" not in await toolset.get_available_tasks(context)

    await toolset.update_task_status(context, "first", TaskStatus.completed)
    assert "second" in await toolset.get_available_tasks(context)


def test_tool_allowlist_changes_tools_and_guidance_together() -> None:
    capability = Planning.from_spec(tools=["write_plan"])
    toolset = cast(PlanningToolset[Any], capability.get_toolset())
    guidance = cast(str | None, capability.get_instructions())

    assert set(toolset.tools) == {"write_plan"}
    assert "write_plan" in (guidance or "")
    assert "read_plan" not in (guidance or "")


def test_agent_spec_storage_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="database or path"):
        Planning.from_spec(database="/tmp/plans.db")

    with pytest.raises(ValueError, match="database or path"):
        Planning.from_spec(backend="sqlite")


def test_planning_capability_serializes_as_one_entry_point() -> None:
    spec = AgentSpec(capabilities=[{"Planning": {"tools": ["write_plan"]}}])
    assert spec.capabilities[0].name == "Planning"
    assert spec.capabilities[0].args == ()
    assert spec.capabilities[0].kwargs == {"tools": ["write_plan"]}


class PlanningEntryPoint:
    name = "Planning"
    value = "yolop_deep:Planning"
    dist = None

    @staticmethod
    def load() -> type[Planning]:
        return Planning


def test_deep_coding_preset_is_explicit_and_catalog_validated() -> None:
    from yolop_deep import load_deep_coding_spec

    catalog = ProviderCatalog.from_installed()
    spec = load_deep_coding_spec(catalog=catalog)

    assert spec.name == "deep-coding"
    assert [capability.name for capability in spec.capabilities] == [
        "Context",
        "Compaction",
        "Planning",
        "StuckLoop",
        "WarnNearLimits",
        "Workspace",
    ]
    assert spec.instructions is not None
    assert "deep coding agent" in str(spec.instructions)

    planning_only = ProviderCatalog.from_entry_points(
        capability_entry_points=[PlanningEntryPoint()]
    )
    with pytest.raises(ValueError, match="not in provider catalog"):
        load_deep_coding_spec(catalog=planning_only)
