import asyncio

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RunUsage
from pytest import raises
from yolop_postgres_runtime import PostgresRuntimeStore
from yolop_runtime import (
    ExecutionPin,
    RunRelation,
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


async def test_postgres_session_can_checkout_a_terminal_run(postgres_dsn: str) -> None:
    store = await PostgresRuntimeStore(postgres_dsn).open()
    try:
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        first = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="first",
            prompt="First",
        )
        first_claim = await store.claim_run(
            "tenant/acme",
            first.run.id,
            owner_id="worker-1",
            lease_seconds=30,
        )
        first_messages = [
            ModelRequest(parts=[UserPromptPart("First")]),
            ModelResponse(parts=[TextPart("First answer")]),
        ]
        first_completion = await store.complete_run(
            "tenant/acme",
            first_claim.id,
            owner_id="worker-1",
            expected_session_revision=session.revision,
            messages=first_messages,
            output="First answer",
            usage=RunUsage(requests=1),
        )
        second = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="second",
            prompt="Second",
        )
        second_claim = await store.claim_run(
            "tenant/acme",
            second.run.id,
            owner_id="worker-2",
            lease_seconds=30,
        )
        second_completion = await store.complete_run(
            "tenant/acme",
            second_claim.id,
            owner_id="worker-2",
            expected_session_revision=first_completion.session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("Second")])],
            output="Second answer",
            usage=RunUsage(requests=1),
        )

        checked_out = await store.checkout_session(
            "tenant/acme",
            session.id,
            first.run.id,
            expected_revision=second_completion.session.revision,
        )

        assert checked_out.head_run_id == first.run.id
        assert checked_out.messages == first_messages
    finally:
        await store.close()


async def test_postgres_session_delete_cascades_run_tree(postgres_dsn: str) -> None:
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
        )
        root_claim = await store.claim_run(
            "tenant/acme",
            root.run.id,
            owner_id="worker-1",
            lease_seconds=30,
        )
        completion = await store.complete_run(
            "tenant/acme",
            root_claim.id,
            owner_id="worker-1",
            expected_session_revision=session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("Root")])],
            output="Done",
            usage=RunUsage(requests=1),
        )
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="child",
            prompt="Child",
            parent_run_id=root.run.id,
            relation=RunRelation.CHILD,
        )

        await store.delete_session(
            "tenant/acme",
            session.id,
            expected_revision=completion.session.revision,
        )

        with raises(SessionNotFoundError):
            await store.load_session("tenant/acme", session.id)
        assert await store.list_runs("tenant/acme", session_id=session.id) == []
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
