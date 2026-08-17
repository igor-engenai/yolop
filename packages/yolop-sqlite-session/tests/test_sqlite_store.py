import asyncio
import sqlite3
import threading
from collections.abc import Sequence

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pytest import raises
from yolop_runtime import (
    ExecutionPin,
    RuntimeStoreSchemaError,
    SessionConflictError,
    SessionFormatError,
    SessionLockTimeoutError,
    SessionNotFoundError,
)
from yolop_sqlite_session import SQLiteRuntimeStore


def execution_pin() -> ExecutionPin:
    return ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")


async def test_runtime_sessions_are_isolated_by_namespace(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")

    acme = await store.create_session("tenant/acme", pin=execution_pin())
    beta = await store.create_session("tenant/beta", pin=execution_pin())

    assert await store.list_sessions("tenant/acme") == [acme.id]
    assert await store.list_sessions("tenant/beta") == [beta.id]
    with raises(SessionNotFoundError):
        await store.load_session("tenant/beta", acme.id)


async def test_runtime_session_pin_and_messages_persist(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    session = await store.create_session("tenant/acme", pin=execution_pin())
    messages = [ModelRequest(parts=[UserPromptPart("Question")])]

    saved = await store.replace_session(
        "tenant/acme",
        session.id,
        expected_revision=session.revision,
        messages=messages,
    )
    loaded = await SQLiteRuntimeStore(database).load_session("tenant/acme", session.id)

    assert loaded == saved
    assert loaded.pin == execution_pin()
    assert loaded.messages == messages


async def test_runtime_session_replace_uses_revision_cas(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session("tenant/acme", pin=execution_pin())

    results = await asyncio.gather(
        store.replace_session(
            "tenant/acme",
            session.id,
            expected_revision=session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("First")])],
        ),
        store.replace_session(
            "tenant/acme",
            session.id,
            expected_revision=session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("Second")])],
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, SessionConflictError) for result in results) == 1


async def test_cancelled_mutation_waits_for_its_worker_before_releasing(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowStore(SQLiteRuntimeStore):
        def _replace_runtime_session(
            self,
            namespace: str,
            session_id: str,
            expected_revision: str,
            messages: Sequence[ModelMessage],
        ):
            started.set()
            release.wait()
            return super()._replace_runtime_session(
                namespace,
                session_id,
                expected_revision,
                messages,
            )

    store = SlowStore(tmp_path / "runtime.db")
    session = await store.create_session("tenant/acme", pin=execution_pin())
    messages = [ModelRequest(parts=[UserPromptPart("Saved")])]
    mutation = asyncio.create_task(
        store.replace_session(
            "tenant/acme",
            session.id,
            expected_revision=session.revision,
            messages=messages,
        )
    )
    try:
        await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
        mutation.cancel()
        await asyncio.sleep(0.02)
        assert mutation.done() is False
        release.set()
        with raises(asyncio.CancelledError):
            await mutation
        assert (await store.load_session("tenant/acme", session.id)).messages == messages
    finally:
        release.set()
        await asyncio.gather(mutation, return_exceptions=True)


async def test_runtime_session_delete_uses_namespace_and_revision(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session("tenant/acme", pin=execution_pin())
    updated = await store.replace_session(
        "tenant/acme",
        session.id,
        expected_revision=session.revision,
        messages=[ModelRequest(parts=[UserPromptPart("Saved")])],
    )

    with raises(SessionConflictError):
        await store.delete_session(
            "tenant/acme",
            session.id,
            expected_revision=session.revision,
        )
    await store.delete_session(
        "tenant/acme",
        session.id,
        expected_revision=updated.revision,
    )
    with raises(SessionNotFoundError):
        await store.load_session("tenant/acme", session.id)


async def test_runtime_session_lock_coordinates_store_instances(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    first = SQLiteRuntimeStore(database)
    second = SQLiteRuntimeStore(database)
    session = await first.create_session("tenant/acme", pin=execution_pin())
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_first_lock() -> None:
        async with first.lock_session("tenant/acme", session.id, timeout=1):
            entered.set()
            await release.wait()

    holder = asyncio.create_task(hold_first_lock())
    await entered.wait()
    try:
        with raises(SessionLockTimeoutError):
            async with second.lock_session("tenant/acme", session.id, timeout=0.01):
                pass
    finally:
        release.set()
        await holder

    async with second.lock_session("tenant/acme", session.id, timeout=1):
        pass


def test_runtime_store_rejects_the_previous_runtime_schema(tmp_path) -> None:
    database = tmp_path / "previous.db"
    store = SQLiteRuntimeStore(database)
    del store
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE runtime_metadata SET schema_version = 1")

    with raises(RuntimeStoreSchemaError):
        SQLiteRuntimeStore(database)


def test_runtime_store_rejects_the_old_schema(tmp_path) -> None:
    database = tmp_path / "old.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, revision TEXT, messages BLOB)"
        )

    with raises(RuntimeStoreSchemaError) as raised:
        SQLiteRuntimeStore(database)

    assert raised.value.code == "runtime_store_schema_mismatch"


async def test_runtime_session_create_fails_fast_for_other_constraints(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_runtime_session
            BEFORE INSERT ON runtime_sessions
            BEGIN
                SELECT RAISE(ABORT, 'rejected');
            END
            """
        )

    with raises(sqlite3.IntegrityError) as raised:
        await store.create_session("tenant/acme", pin=execution_pin())

    assert raised.value.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_TRIGGER


async def test_load_reports_malformed_runtime_messages(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    session = await store.create_session("tenant/acme", pin=execution_pin())
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE runtime_sessions SET messages = ?
            WHERE namespace = ? AND id = ?
            """,
            (b"not-json", "tenant/acme", session.id),
        )

    with raises(SessionFormatError, match=session.id):
        await store.load_session("tenant/acme", session.id)
