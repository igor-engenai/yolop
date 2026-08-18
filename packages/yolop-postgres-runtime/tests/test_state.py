from __future__ import annotations

import asyncio

from yolop_postgres_runtime import PostgresRuntimeStore
from yolop_runtime import (
    ExecutionPin,
    StateScope,
    StateSequenceConflictError,
)


async def test_postgres_scoped_state_is_bounded_and_compare_append_is_atomic(
    postgres_dsn: str,
) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        first = await store.append_state(
            "tenant/acme",
            owner_id="plugin.counter",
            scope=StateScope.SESSION,
            scope_id=session.id,
            state_kind="counter",
            schema_version=1,
            expected_sequence=0,
            payload={"count": 1},
        )

        async def append(count: int):
            return await store.append_state(
                "tenant/acme",
                owner_id="plugin.counter",
                scope=StateScope.SESSION,
                scope_id=session.id,
                state_kind="counter",
                schema_version=1,
                expected_sequence=first.sequence,
                payload={"count": count},
            )

        results = await asyncio.gather(append(2), append(3), return_exceptions=True)
        entries = await store.read_state(
            "tenant/acme",
            owner_id="plugin.counter",
            scope=StateScope.SESSION,
            scope_id=session.id,
            state_kind="counter",
            schema_version=1,
        )

        assert sum(isinstance(result, StateSequenceConflictError) for result in results) == 1
        assert [entry.sequence for entry in entries] == [1, 2]
        assert entries[0] == first
        assert len(entries) == 2
        assert entries[1].payload["count"] in {2, 3}
    finally:
        await store.close()


async def test_postgres_scoped_state_isolated_by_namespace(postgres_dsn: str) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        await store.append_state(
            "tenant/acme",
            owner_id="plugin.counter",
            scope=StateScope.SESSION,
            scope_id=session.id,
            state_kind="counter",
            schema_version=1,
            expected_sequence=0,
            payload={"count": 1},
        )

        assert await store.read_state(
            "tenant/beta",
            owner_id="plugin.counter",
            scope=StateScope.SESSION,
            scope_id=session.id,
            state_kind="counter",
            schema_version=1,
        ) == []
    finally:
        await store.close()
