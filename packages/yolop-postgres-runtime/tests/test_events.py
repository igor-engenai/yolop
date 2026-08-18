from __future__ import annotations

import asyncio

from pytest import raises
from yolop_postgres_runtime import PostgresRuntimeStore
from yolop_runtime import ExecutionPin, RunStateError


async def test_postgres_run_events_are_ordered_and_owner_bound(postgres_dsn: str) -> None:
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
            await store.append_run_event(
                "tenant/acme",
                claimed.id,
                owner_id="worker-2",
                event="wrong_owner",
                data="{}",
            )
        first = await store.append_run_event(
            "tenant/acme",
            claimed.id,
            owner_id="worker-1",
            event="tool_start",
            data='{"name":"lookup"}',
        )
        second = await store.append_run_event(
            "tenant/acme",
            claimed.id,
            owner_id="worker-1",
            event="tool_end",
            data='{"ok":true}',
        )

        assert await store.list_run_events("tenant/acme", claimed.id) == [first, second]
        assert await store.list_run_events("tenant/acme", claimed.id, after=first.sequence) == [
            second
        ]
        assert (await store.load_run("tenant/acme", claimed.id)).events == [first, second]
    finally:
        await store.close()


async def test_postgres_concurrent_event_appends_have_no_duplicate_sequence(
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

        events = await asyncio.gather(
            *(
                store.append_run_event(
                    "tenant/acme",
                    claimed.id,
                    owner_id="worker-1",
                    event="delta",
                    data=f'{{"index":{index}}}',
                )
                for index in range(5)
            )
        )
        persisted = await store.list_run_events("tenant/acme", claimed.id)

        assert sorted(event.sequence for event in events) == [1, 2, 3, 4, 5]
        assert [event.sequence for event in persisted] == [1, 2, 3, 4, 5]
    finally:
        await store.close()


async def test_postgres_event_notification_is_only_a_wakeup(postgres_dsn: str) -> None:
    publisher = await PostgresRuntimeStore(postgres_dsn).open()
    listener = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await publisher.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        reservation = await publisher.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request-1",
            prompt="Hello",
        )
        claimed = await publisher.claim_run(
            "tenant/acme",
            reservation.run.id,
            owner_id="worker-1",
            lease_seconds=30,
        )

        async with listener.event_notifications() as notifications:
            await publisher.append_run_event(
                "tenant/acme",
                claimed.id,
                owner_id="worker-1",
                event="completed",
                data='{"ok":true}',
            )
            notification = await asyncio.wait_for(anext(notifications), timeout=5)

        assert notification.payload.endswith(":1")
        assert len(await listener.list_run_events("tenant/acme", claimed.id)) == 1
    finally:
        await publisher.close()
        await listener.close()
