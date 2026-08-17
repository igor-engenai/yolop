from __future__ import annotations

from pydantic_ai import AgentSpec
from pydantic_ai.models.test import TestModel
from yolop_deep import Planning, SessionPlanStore
from yolop_runtime import Runtime, ScopedStateContext
from yolop_sqlite_session import SQLiteRuntimeStore

from yolop import ProviderCatalog


class EntryPoint:
    name = "Planning"
    value = "yolop_deep:Planning"
    dist = None

    @staticmethod
    def load() -> type[Planning]:
        return Planning


async def test_planning_capability_creates_and_reads_a_plan_in_one_run(tmp_path) -> None:
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
