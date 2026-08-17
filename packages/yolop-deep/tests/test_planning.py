from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from yolop_deep import PlanItem, Planning, SessionPlanStore
from yolop_runtime import Runtime, ScopedStateContext, StateSequenceConflictError
from yolop_sqlite_session import SQLiteRuntimeStore

from yolop import ProviderCatalog


class EntryPoint:
    name = "Planning"
    value = "yolop_deep:Planning"
    dist = None

    @staticmethod
    def load() -> type[Planning]:
        return Planning


async def test_planning_capability_creates_and_reads_a_plan_in_one_run(tmp_path: Path) -> None:
    catalog = ProviderCatalog.from_entry_points(capability_entry_points=[EntryPoint()])
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store, provider_catalog=catalog)
    spec = AgentSpec(model="test:model", capabilities=[{"Planning": {}}])
    session = await runtime.create_session("tenant", spec=spec, model_id="test:model")

    await runtime.run(
        "tenant",
        session.id,
        "Make a plan for the work.",
        spec=spec,
        model=TestModel(call_tools=["write_plan", "read_plan"]),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="plan-once",
    )

    plan = SessionPlanStore(
        ScopedStateContext(
            store=store,
            namespace="tenant",
            session_id=session.id,
            run_id=session.id,
        )
    )
    items = await plan.get_items()
    assert items
    assert all(item.content for item in items)


def _tool_then_text(name: str, arguments: dict[str, object], seen: list[str]) -> FunctionModel:
    async def respond(
        messages: list[ModelMessage], _info: object
    ) -> AsyncIterator[str | DeltaToolCalls]:
        returns = (
            [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
            if isinstance(messages[-1], ModelRequest)
            else []
        )
        if returns:
            yield str(returns[-1].content)
            return
        seen.append(name)
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(arguments), tool_call_id=name)}

    return FunctionModel(stream_function=respond, model_name="test")


async def test_plan_state_preserves_compare_and_append_conflicts(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    spec = AgentSpec(model="test:model")
    session = await runtime.create_session("tenant", spec=spec, model_id="test:model")
    state = ScopedStateContext(
        store=store,
        namespace="tenant",
        session_id=session.id,
        run_id=session.id,
    )
    plan = SessionPlanStore(state)
    await plan.set_items([PlanItem(content="first")])
    state_stream = state.for_session("yolop.deep.planning")
    current = await state_stream.read("plan")
    await state_stream.append(
        "plan",
        {"items": [PlanItem(content="external").model_dump(mode="json")]},
        expected_sequence=current[-1].sequence,
    )

    with pytest.raises(StateSequenceConflictError):
        await state_stream.append(
            "plan",
            {"items": [PlanItem(content="stale writer").model_dump(mode="json")]},
            expected_sequence=current[-1].sequence,
        )

    assert [item.content for item in await plan.get_items()] == ["external"]


async def test_process_restart_preserves_the_session_plan(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    first_store = SQLiteRuntimeStore(database)
    first_runtime = Runtime(store=first_store)
    spec = AgentSpec(model="test:model")
    session = await first_runtime.create_session("tenant", spec=spec, model_id="test:model")
    first_plan = SessionPlanStore(
        ScopedStateContext(
            store=first_store,
            namespace="tenant",
            session_id=session.id,
            run_id=session.id,
        )
    )
    await first_plan.set_items([PlanItem(content="survive restart")])

    reopened_store = SQLiteRuntimeStore(database)
    reopened_plan = SessionPlanStore(
        ScopedStateContext(
            store=reopened_store,
            namespace="tenant",
            session_id=session.id,
            run_id=session.id,
        )
    )

    assert [item.content for item in await reopened_plan.get_items()] == ["survive restart"]


async def test_plans_are_isolated_by_session_and_namespace(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    spec = AgentSpec(model="test:model")
    first = await runtime.create_session("tenant-a", spec=spec, model_id="test:model")
    second = await runtime.create_session("tenant-a", spec=spec, model_id="test:model")
    other_namespace = await runtime.create_session("tenant-b", spec=spec, model_id="test:model")

    async def plan_for(namespace: str, session_id: str) -> SessionPlanStore:
        return SessionPlanStore(
            ScopedStateContext(
                store=store,
                namespace=namespace,
                session_id=session_id,
                run_id=session_id,
            )
        )

    first_plan = await plan_for("tenant-a", first.id)
    second_plan = await plan_for("tenant-a", second.id)
    other_plan = await plan_for("tenant-b", other_namespace.id)
    await first_plan.set_items([PlanItem(content="first")])
    await second_plan.set_items([PlanItem(content="second")])
    await other_plan.set_items([PlanItem(content="other namespace")])

    assert [item.content for item in await first_plan.get_items()] == ["first"]
    assert [item.content for item in await second_plan.get_items()] == ["second"]
    assert [item.content for item in await other_plan.get_items()] == ["other namespace"]


async def test_later_run_reads_the_same_session_plan(tmp_path: Path) -> None:
    catalog = ProviderCatalog.from_entry_points(capability_entry_points=[EntryPoint()])
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store, provider_catalog=catalog)
    spec = AgentSpec(model="test:model", capabilities=[{"Planning": {}}])
    session = await runtime.create_session("tenant", spec=spec, model_id="test:model")
    first_calls: list[str] = []
    second_calls: list[str] = []

    await runtime.run(
        "tenant",
        session.id,
        "Create the plan.",
        spec=spec,
        model=_tool_then_text(
            "write_plan",
            {"items": [{"content": "persist the plan", "active_form": "Persisting the plan"}]},
            first_calls,
        ),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="plan-write",
    )
    completion = await runtime.run(
        "tenant",
        session.id,
        "Read the plan.",
        spec=spec,
        model=_tool_then_text("read_plan", {}, second_calls),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="plan-read",
    )

    assert first_calls == ["write_plan"]
    assert second_calls == ["read_plan"]
    assert "persist the plan" in str(completion.run.output)
