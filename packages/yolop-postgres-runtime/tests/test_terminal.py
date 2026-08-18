from __future__ import annotations

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RunUsage
from pytest import raises
from yolop_postgres_runtime import PostgresRuntimeStore
from yolop_runtime import (
    ExecutionPin,
    RunRelation,
    RunStateError,
    RunStatus,
    RuntimeBudget,
)


async def test_postgres_completion_updates_session_and_run_atomically(
    postgres_dsn: str,
) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        reservation = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request-1",
            prompt="Hello",
        )
        claimed = await store.claim_run(
            "tenant/acme",
            reservation.run.id,
            owner_id="worker-1",
            lease_seconds=30,
        )
        messages = [
            ModelRequest(parts=[UserPromptPart("Hello")]),
            ModelResponse(parts=[TextPart("Hi")]),
        ]

        completion = await store.complete_run(
            "tenant/acme",
            claimed.id,
            owner_id="worker-1",
            expected_session_revision=session.revision,
            messages=messages,
            output={"answer": "Hi"},
            usage=RunUsage(requests=1, input_tokens=2, output_tokens=1),
        )
        reopened = await PostgresRuntimeStore(postgres_dsn).open()
        try:
            loaded_session = await reopened.load_session("tenant/acme", session.id)
            loaded_run = await reopened.load_run("tenant/acme", claimed.id)
        finally:
            await reopened.close()

        assert completion.run.status is RunStatus.COMPLETED
        assert completion.run.output == {"answer": "Hi"}
        assert completion.run.usage == RunUsage(requests=1, input_tokens=2, output_tokens=1)
        assert loaded_session == completion.session
        assert loaded_run == completion.run
        assert loaded_session.head_run_id == claimed.id
    finally:
        await store.close()


async def test_postgres_terminal_transitions_reject_stale_owners(
    postgres_dsn: str,
) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        reservation = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request-1",
            prompt="Hello",
        )
        claimed = await store.claim_run(
            "tenant/acme",
            reservation.run.id,
            owner_id="worker-1",
            lease_seconds=30,
        )

        with raises(RunStateError):
            await store.fail_run(
                "tenant/acme",
                claimed.id,
                owner_id="worker-2",
                error_code="wrong-owner",
                error_detail="Wrong owner",
            )
        with raises(RunStateError):
            await store.complete_run(
                "tenant/acme",
                claimed.id,
                owner_id="worker-2",
                expected_session_revision=session.revision,
                messages=[],
                output="lost",
                usage=RunUsage(requests=1),
            )

        assert (await store.load_run("tenant/acme", claimed.id)).status is RunStatus.RUNNING
    finally:
        await store.close()


async def test_postgres_cancellation_invalidates_run_ownership(postgres_dsn: str) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        reservation = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request-1",
            prompt="Hello",
        )
        claimed = await store.claim_run(
            "tenant/acme",
            reservation.run.id,
            owner_id="worker-1",
            lease_seconds=30,
        )

        cancelled = await store.cancel_run("tenant/acme", claimed.id)

        assert cancelled.status is RunStatus.INTERRUPTED
        assert cancelled.owner_id is None
        with raises(RunStateError):
            await store.complete_run(
                "tenant/acme",
                claimed.id,
                owner_id="worker-1",
                expected_session_revision=session.revision,
                messages=[],
                output="late",
                usage=RunUsage(requests=1),
            )
        assert await store.cancel_run("tenant/acme", claimed.id) == cancelled
    finally:
        await store.close()


async def test_postgres_terminal_usage_accounts_root_budget(postgres_dsn: str) -> None:
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
            root_budget=RuntimeBudget(request_limit=2, continuation_limit=1),
        )
        root_claim = await store.claim_run(
            "tenant/acme",
            root.run.id,
            owner_id="worker-root",
            lease_seconds=30,
        )
        await store.complete_run(
            "tenant/acme",
            root_claim.id,
            owner_id="worker-root",
            expected_session_revision=session.revision,
            messages=[],
            output="root",
            usage=RunUsage(requests=1, input_tokens=2, output_tokens=3),
        )
        continuation = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="continuation",
            prompt="Continue",
            parent_run_id=root.run.id,
            relation=RunRelation.CONTINUATION,
        )
        continuation_claim = await store.claim_run(
            "tenant/acme",
            continuation.run.id,
            owner_id="worker-continuation",
            lease_seconds=30,
        )
        await store.complete_run(
            "tenant/acme",
            continuation_claim.id,
            owner_id="worker-continuation",
            expected_session_revision=(
                await store.load_session("tenant/acme", session.id)
            ).revision,
            messages=[],
            output="continuation",
            usage=RunUsage(requests=1, input_tokens=4, output_tokens=5),
        )
        budget = await store.load_root_budget("tenant/acme", root.run.id)

        assert budget is not None
        assert budget.requests_used == 2
        assert budget.input_tokens_used == 6
        assert budget.output_tokens_used == 8
        assert budget.total_tokens_used == 14
        assert budget.continuations_used == 1
        assert budget.active_runs == 0
    finally:
        await store.close()
