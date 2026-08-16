import asyncio
import sqlite3
from uuid import UUID

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pytest import raises
from yolop_session import (
    ExecutionPin,
    RuntimeStoreSchemaError,
    SessionConflictError,
    SessionFormatError,
    SessionLockTimeoutError,
    SessionNotFoundError,
    SessionSnapshot,
)
from yolop_sqlite_session import SQLiteRuntimeStore, SQLiteSessionStore


async def test_runtime_sessions_are_isolated_by_namespace(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")

    acme = await store.create_session("tenant/acme", pin=pin)
    beta = await store.create_session("tenant/beta", pin=pin)

    assert await store.list_sessions("tenant/acme") == [acme.id]
    assert await store.list_sessions("tenant/beta") == [beta.id]
    with raises(SessionNotFoundError):
        await store.load_session("tenant/beta", acme.id)


async def test_runtime_session_pin_and_messages_persist(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    store = SQLiteRuntimeStore(database)
    session = await store.create_session("tenant/acme", pin=pin)
    messages = [ModelRequest(parts=[UserPromptPart("Question")])]

    saved = await store.replace_session(
        "tenant/acme",
        session.id,
        expected_revision=session.revision,
        messages=messages,
    )
    loaded = await SQLiteRuntimeStore(database).load_session("tenant/acme", session.id)

    assert loaded == saved
    assert loaded.pin == pin
    assert loaded.messages == messages


async def test_runtime_session_replace_uses_revision_cas(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)

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


async def test_runtime_session_delete_uses_namespace_and_revision(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)
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
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await first.create_session("tenant/acme", pin=pin)
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
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")

    with raises(sqlite3.IntegrityError) as raised:
        await store.create_session("tenant/acme", pin=pin)

    assert raised.value.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_TRIGGER


async def test_created_session_persists_across_store_instances(tmp_path) -> None:
    database = tmp_path / "sessions.db"
    session = await SQLiteSessionStore(database).create()

    assert UUID(session.id).version == 4
    assert session.messages == []
    assert session.revision
    assert await SQLiteSessionStore(database).list_sessions() == [session.id]


async def test_replace_and_load_preserve_native_messages(tmp_path) -> None:
    database = tmp_path / "sessions.db"
    store = SQLiteSessionStore(database)
    session = await store.create()
    messages = [
        ModelRequest(parts=[UserPromptPart("Question")]),
        ModelResponse(parts=[TextPart("Answer")]),
    ]

    updated = await store.replace(
        session.id,
        expected_revision=session.revision,
        messages=messages,
    )
    loaded = await SQLiteSessionStore(database).load(session.id)

    assert updated.revision != session.revision
    assert loaded == updated
    assert loaded.messages == messages


async def test_only_one_concurrent_replace_can_use_a_revision(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session = await store.create()

    results = await asyncio.gather(
        store.replace(
            session.id,
            expected_revision=session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("First")])] * 1_000,
        ),
        store.replace(
            session.id,
            expected_revision=session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("Second")])] * 1_000,
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, SessionSnapshot) for result in results) == 1
    assert sum(isinstance(result, SessionConflictError) for result in results) == 1


async def test_delete_requires_the_current_revision(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    session = await store.create()
    updated = await store.replace(
        session.id,
        expected_revision=session.revision,
        messages=[ModelRequest(parts=[UserPromptPart("Saved")])],
    )

    with raises(SessionConflictError, match=session.id):
        await store.delete(session.id, expected_revision=session.revision)

    await store.delete(session.id, expected_revision=updated.revision)
    with raises(SessionNotFoundError, match=session.id):
        await store.load(session.id)


async def test_load_reports_malformed_stored_messages(tmp_path) -> None:
    database = tmp_path / "sessions.db"
    store = SQLiteSessionStore(database)
    session = await store.create()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE sessions SET messages = ? WHERE id = ?",
            (b"not-json", session.id),
        )

    with raises(SessionFormatError, match=session.id):
        await store.load(session.id)
