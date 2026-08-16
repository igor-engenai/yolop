import asyncio
import sqlite3
from uuid import UUID

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pytest import raises
from yolop_session import (
    SessionConflictError,
    SessionFormatError,
    SessionNotFoundError,
    SessionSnapshot,
)
from yolop_sqlite_session import SQLiteSessionStore


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
