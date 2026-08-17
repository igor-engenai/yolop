from __future__ import annotations

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.models.test import TestModel
from yolop_runtime import RunRelation, Runtime, SessionConflictError
from yolop_sqlite_session import SQLiteRuntimeStore


async def test_checkout_and_fork_preserve_exact_history_and_pins(tmp_path) -> None:
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
    source_full = list(second.run.full_messages)
    before_checkout = await runtime.load_session("tenant", session.id)

    checked_out = await runtime.checkout(
        "tenant",
        session.id,
        first.run.id,
        expected_revision=before_checkout.revision,
    )

    assert checked_out.head_run_id == first.run.id
    assert checked_out.messages == first.run.active_messages
    assert second.run.full_messages == source_full

    with pytest.raises(SessionConflictError):
        await runtime.checkout(
            "tenant",
            session.id,
            second.run.id,
            expected_revision=before_checkout.revision,
        )
    unchanged = await runtime.load_session("tenant", session.id)
    assert unchanged.head_run_id == first.run.id

    forked = await runtime.fork_session(
        "tenant",
        session.id,
        second.run.id,
        expected_revision=unchanged.revision,
    )
    assert forked.id != session.id
    assert forked.pin == session.pin
    assert forked.messages == second.run.active_messages
    assert await runtime.list_runs("tenant", session_id=session.id)
    assert await runtime.list_runs("tenant", session_id=forked.id) == []
