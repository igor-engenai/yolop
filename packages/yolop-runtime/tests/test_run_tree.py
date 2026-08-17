from __future__ import annotations

from pydantic_ai import AgentSpec
from pydantic_ai.models.test import TestModel
from yolop_runtime import RunRelation, Runtime, RunTreeNode
from yolop_sqlite_session import SQLiteRuntimeStore


async def test_runtime_lists_linear_and_branched_run_tree(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    spec = AgentSpec(model="test:model")
    session = await runtime.create_session("tenant", spec=spec, model_id="test:model")

    first = await runtime.run(
        "tenant",
        session.id,
        "first",
        spec=spec,
        model=TestModel(custom_output_text="one"),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="first",
    )
    second = await runtime.run_related(
        "tenant",
        session.id,
        "second",
        parent_run_id=first.run.id,
        relation=RunRelation.CONTINUATION,
        spec=spec,
        model=TestModel(custom_output_text="two"),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="second",
    )
    current = await runtime.load_session("tenant", session.id)
    await runtime.checkout(
        "tenant",
        session.id,
        first.run.id,
        expected_revision=current.revision,
    )
    await runtime.run_related(
        "tenant",
        session.id,
        "third",
        parent_run_id=first.run.id,
        relation=RunRelation.CONTINUATION,
        spec=spec,
        model=TestModel(custom_output_text="three"),
        model_id="test:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="third",
    )

    tree = await runtime.list_run_tree("tenant", session_id=session.id)

    assert len(tree) == 1
    assert isinstance(tree[0], RunTreeNode)
    assert tree[0].run.id == first.run.id
    assert [child.run.id for child in tree[0].children] == [
        second.run.id,
        tree[0].children[1].run.id,
    ]
    assert {child.run.prompt for child in tree[0].children} == {"second", "third"}
