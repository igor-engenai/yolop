from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pytest import raises
from yolop_postgres_runtime import PostgresRuntimeStore
from yolop_runtime import (
    ExecutionPin,
    IdempotencyConflictError,
    RunAdmissionError,
    RunBudgetExceededError,
    RunRelation,
    RunStatus,
    RuntimeBudget,
)


async def test_postgres_concurrent_reservation_is_idempotent(postgres_dsn: str) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )

        async def reserve():
            return await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="request-1",
                prompt="Hello",
            )

        first, second = await asyncio.gather(reserve(), reserve())

        assert first.run.id == second.run.id
        assert sorted((first.created, second.created)) == [False, True]
        assert first.run.status is RunStatus.ACCEPTED
    finally:
        await store.close()


async def test_postgres_idempotency_rejects_different_input(postgres_dsn: str) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request-1",
            prompt="First",
        )

        with raises(IdempotencyConflictError):
            await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="request-1",
                prompt="Different",
            )
    finally:
        await store.close()


async def test_postgres_reservation_enforces_per_session_pending_limit(
    postgres_dsn: str,
) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        first = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request-1",
            prompt="First",
            max_pending=1,
        )

        with raises(RunAdmissionError):
            await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="request-2",
                prompt="Second",
                max_pending=1,
            )
        replay = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request-1",
            prompt="First",
            max_pending=1,
        )

        assert replay.created is False
        assert replay.run.id == first.run.id
    finally:
        await store.close()


async def test_postgres_run_relations_and_message_ranges_are_durable(
    postgres_dsn: str,
) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        root_messages = [ModelRequest(parts=[UserPromptPart("Root")])]
        root = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="root",
            prompt="Root",
            full_messages=root_messages,
            active_messages=root_messages,
        )
        child = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="child",
            prompt="Child",
            parent_run_id=root.run.id,
            relation=RunRelation.CHILD,
            full_messages=[ModelRequest(parts=[UserPromptPart("Child")])],
        )
        continuation = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="continuation",
            prompt="Continue",
            parent_run_id=root.run.id,
            relation=RunRelation.CONTINUATION,
        )
        loaded_root = await store.load_run("tenant/acme", root.run.id)
        listed = await store.list_runs("tenant/acme", session_id=session.id)

        assert isinstance(loaded_root.full_messages[0], ModelRequest)
        assert isinstance(loaded_root.active_messages[0], ModelRequest)
        full_part = loaded_root.full_messages[0].parts[0]
        active_part = loaded_root.active_messages[0].parts[0]
        assert isinstance(full_part, UserPromptPart)
        assert isinstance(active_part, UserPromptPart)
        assert full_part.content == "Root"
        assert active_part.content == "Root"
        assert child.run.parent_run_id == root.run.id
        assert child.run.root_run_id == root.run.id
        assert child.run.relation is RunRelation.CHILD
        assert continuation.run.root_run_id == root.run.id
        assert continuation.run.relation is RunRelation.CONTINUATION
        assert {run.id for run in listed} == {
            root.run.id,
            child.run.id,
            continuation.run.id,
        }
    finally:
        await store.close()


async def test_postgres_root_budget_limits_related_runs_and_reloads(
    postgres_dsn: str,
) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        root = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="root",
            prompt="Root",
            root_budget=RuntimeBudget(
                request_limit=3,
                child_run_limit=1,
                continuation_limit=1,
            ),
        )
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="child",
            prompt="Child",
            parent_run_id=root.run.id,
            relation=RunRelation.CHILD,
        )
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="continuation",
            prompt="Continue",
            parent_run_id=root.run.id,
            relation=RunRelation.CONTINUATION,
        )

        with raises(RunBudgetExceededError):
            await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="second-child",
                prompt="Second child",
                parent_run_id=root.run.id,
                relation=RunRelation.CHILD,
            )
        budget = await store.load_root_budget("tenant/acme", root.run.id)

        assert budget is not None
        assert budget.budget.request_limit == 3
        assert budget.child_runs_used == 1
        assert budget.continuations_used == 1
        assert budget.active_runs == 3
    finally:
        await store.close()


async def test_postgres_root_budget_rejects_expired_wall_deadline(postgres_dsn: str) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )

        with raises(RunBudgetExceededError):
            await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="expired",
                prompt="Expired",
                root_budget=RuntimeBudget(
                    wall_deadline=datetime.now(UTC) - timedelta(seconds=1),
                ),
            )
    finally:
        await store.close()
