from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from yolop_session import (
    SessionConflictError,
    SessionFormatError,
    SessionNotFoundError,
    SessionSnapshot,
    new_session_id,
    validate_session_id,
)

_EMPTY_MESSAGES = b"[]"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    revision TEXT NOT NULL,
    messages BLOB NOT NULL
)
"""


class SQLiteSessionStore:
    """Store agent sessions in a host-provided SQLite database."""

    def __init__(self, database: str | Path) -> None:
        self._database = Path(database).expanduser().resolve()
        self._database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(_SCHEMA)

    async def create(self) -> SessionSnapshot:
        """Create and return an empty session with a generated ID."""
        return await asyncio.to_thread(self._create)

    async def list_sessions(self) -> list[str]:
        """List session IDs in stable order."""
        return await asyncio.to_thread(self._list_sessions)

    async def load(self, session_id: str) -> SessionSnapshot:
        """Load one session and its content revision."""
        return await asyncio.to_thread(self._load, session_id)

    async def delete(self, session_id: str, *, expected_revision: str) -> None:
        """Delete a session if its revision is current."""
        await asyncio.to_thread(self._delete, session_id, expected_revision)

    async def replace(
        self,
        session_id: str,
        *,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> SessionSnapshot:
        """Atomically replace a session's complete message history."""
        return await asyncio.to_thread(
            self._replace,
            session_id,
            expected_revision,
            messages,
        )

    def _create(self) -> SessionSnapshot:
        while True:
            session_id = new_session_id()
            revision = _revision(_EMPTY_MESSAGES)
            try:
                with self._connect() as connection:
                    connection.execute(
                        "INSERT INTO sessions (id, revision, messages) VALUES (?, ?, ?)",
                        (session_id, revision, _EMPTY_MESSAGES),
                    )
            except sqlite3.IntegrityError:
                continue
            return SessionSnapshot(id=session_id, messages=[], revision=revision)

    def _list_sessions(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM sessions ORDER BY id").fetchall()
        return [row[0] for row in rows]

    def _load(self, session_id: str) -> SessionSnapshot:
        validated_id = validate_session_id(session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT revision, messages FROM sessions WHERE id = ?",
                (validated_id,),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session {session_id!r} does not exist")
        revision, content = row
        try:
            messages = ModelMessagesTypeAdapter.validate_json(content)
        except (ValueError, ValidationError) as error:
            raise SessionFormatError(f"Session {session_id!r} has invalid messages") from error
        return SessionSnapshot(id=session_id, messages=messages, revision=revision)

    def _delete(self, session_id: str, expected_revision: str) -> None:
        validated_id = validate_session_id(session_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision FROM sessions WHERE id = ?",
                (validated_id,),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session {session_id!r} does not exist")
            if row[0] != expected_revision:
                raise SessionConflictError(f"Session {session_id!r} has changed")
            connection.execute("DELETE FROM sessions WHERE id = ?", (validated_id,))

    def _replace(
        self,
        session_id: str,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> SessionSnapshot:
        validated_id = validate_session_id(session_id)
        content = ModelMessagesTypeAdapter.dump_json(list(messages))
        revision = _revision(content)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision FROM sessions WHERE id = ?",
                (validated_id,),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session {session_id!r} does not exist")
            if row[0] != expected_revision:
                raise SessionConflictError(f"Session {session_id!r} has changed")
            connection.execute(
                "UPDATE sessions SET revision = ?, messages = ? WHERE id = ?",
                (revision, content, validated_id),
            )
        return SessionSnapshot(id=session_id, messages=list(messages), revision=revision)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, timeout=5)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection


def _revision(content: bytes) -> str:
    return sha256(content).hexdigest()


__all__ = ["SQLiteSessionStore"]
