from __future__ import annotations

import asyncio

from pytest import raises
from yolop_postgres_runtime import PostgresRuntimeStore
from yolop_runtime import ExecutionPin, RunStateError, RunStatus


async def test_postgres_run_can_be_claimed_with_a_lease(postgres_dsn: str) -> None:
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

        assert claimed.status is RunStatus.RUNNING
        assert claimed.owner_id == "worker-1"
        assert claimed.lease_expires_at is not None
    finally:
        await store.close()


async def test_postgres_claim_contention_and_renewal_are_owner_bound(
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
            lease_seconds=1,
        )

        with raises(RunStateError):
            await store.claim_run(
                "tenant/acme",
                reservation.run.id,
                owner_id="worker-2",
                lease_seconds=1,
            )
        with raises(RunStateError):
            await store.renew_run_lease(
                "tenant/acme",
                claimed.id,
                owner_id="worker-2",
                lease_seconds=30,
            )
        renewed = await store.renew_run_lease(
            "tenant/acme",
            claimed.id,
            owner_id="worker-1",
            lease_seconds=30,
        )

        assert renewed.owner_id == "worker-1"
        assert renewed.lease_expires_at is not None
        assert claimed.lease_expires_at is not None
        assert renewed.lease_expires_at > claimed.lease_expires_at
    finally:
        await store.close()


async def test_postgres_shutdown_interrupts_owned_runs(postgres_dsn: str) -> None:
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
        await store.claim_run(
            "tenant/acme",
            reservation.run.id,
            owner_id="worker-1",
            lease_seconds=30,
        )

        assert await store.interrupt_owned_runs("worker-1") == 1
        interrupted = await store.load_run("tenant/acme", reservation.run.id)

        assert interrupted.status is RunStatus.INTERRUPTED
        with raises(RunStateError):
            await store.claim_run(
                "tenant/acme",
                interrupted.id,
                owner_id="worker-2",
                lease_seconds=30,
            )
    finally:
        await store.close()


async def test_postgres_expired_lease_is_interrupted(postgres_dsn: str) -> None:
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
        await store.claim_run(
            "tenant/acme",
            reservation.run.id,
            owner_id="worker-1",
            lease_seconds=0.01,
        )
        await asyncio.sleep(0.02)

        assert await store.interrupt_expired_runs() == 1
        assert (
            await store.load_run("tenant/acme", reservation.run.id)
        ).status is RunStatus.INTERRUPTED
    finally:
        await store.close()
