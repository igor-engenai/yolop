import asyncio

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pytest import raises
from yolop_postgres_runtime import PostgresRuntimeStore
from yolop_runtime import (
    ExecutionPin,
    SessionConflictError,
    SessionLockTimeoutError,
    SessionNotFoundError,
)


async def test_postgres_sessions_are_isolated_by_namespace(postgres_dsn: str) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        acme = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        beta = await store.create_session(
            "tenant/beta",
            pin=ExecutionPin(agent_spec_id="b" * 64, model_id="openai:model"),
        )

        assert await store.list_sessions("tenant/acme") == [acme.id]
        assert await store.list_sessions("tenant/beta") == [beta.id]
        with raises(SessionNotFoundError):
            await store.load_session("tenant/beta", acme.id)
    finally:
        await store.close()


async def test_postgres_session_lock_coordinates_store_instances(postgres_dsn: str) -> None:
    first = await PostgresRuntimeStore(postgres_dsn).open()
    second = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await first.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def hold_lock() -> None:
            async with first.lock_session("tenant/acme", session.id, timeout=1):
                entered.set()
                await release.wait()

        holder = asyncio.create_task(hold_lock())
        await entered.wait()
        try:
            with raises(SessionLockTimeoutError):
                async with second.lock_session("tenant/acme", session.id, timeout=0.02):
                    pass
        finally:
            release.set()
            await holder

        async with second.lock_session("tenant/acme", session.id, timeout=1):
            pass
    finally:
        await first.close()
        await second.close()


async def test_postgres_session_revision_and_pin_are_durable(postgres_dsn: str) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        messages = [ModelRequest(parts=[UserPromptPart("Question")])]

        saved = await store.replace_session(
            "tenant/acme",
            session.id,
            expected_revision=session.revision,
            messages=messages,
        )
        loaded = await store.load_session("tenant/acme", session.id)

        assert loaded == saved
        assert loaded.pin == session.pin
        assert isinstance(loaded.messages[0].parts[0], UserPromptPart)
        assert loaded.messages[0].parts[0].content == "Question"

        with raises(SessionConflictError):
            await store.replace_session(
                "tenant/acme",
                session.id,
                expected_revision=session.revision,
                messages=[],
            )
    finally:
        await store.close()
