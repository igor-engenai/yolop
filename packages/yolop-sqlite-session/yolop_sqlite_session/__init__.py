from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path

from filelock import AsyncFileLock
from filelock import Timeout as FileLockTimeout
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from yolop_session import (
    ExecutionPin,
    RuntimeSessionSnapshot,
    RuntimeStoreSchemaError,
    SessionConflictError,
    SessionFormatError,
    SessionLockTimeoutError,
    SessionNotFoundError,
    SessionSnapshot,
    new_session_id,
    validate_namespace,
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
_RUNTIME_SCHEMA_VERSION = 1
_RUNTIME_METADATA_SCHEMA = """
CREATE TABLE runtime_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL
)
"""
_RUNTIME_SCHEMA = """
CREATE TABLE runtime_sessions (
    namespace TEXT NOT NULL,
    id TEXT NOT NULL,
    agent_spec_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    messages BLOB NOT NULL,
    PRIMARY KEY (namespace, id)
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


class SQLiteRuntimeStore:
    """Store namespaced YoloP runtime state in SQLite."""

    def __init__(self, database: str | Path) -> None:
        self._database = Path(database).expanduser().resolve()
        self._database.parent.mkdir(parents=True, exist_ok=True)
        self._locks_directory = self._database.parent / f".{self._database.name}.locks"
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if tables and "runtime_metadata" not in tables:
                raise RuntimeStoreSchemaError(
                    "Database contains an unsupported pre-runtime schema"
                )
            if not tables:
                connection.execute(_RUNTIME_METADATA_SCHEMA)
                connection.execute(
                    "INSERT INTO runtime_metadata (singleton, schema_version) VALUES (1, ?)",
                    (_RUNTIME_SCHEMA_VERSION,),
                )
                connection.execute(_RUNTIME_SCHEMA)
            else:
                row = connection.execute(
                    "SELECT schema_version FROM runtime_metadata WHERE singleton = 1"
                ).fetchone()
                if row != (_RUNTIME_SCHEMA_VERSION,):
                    raise RuntimeStoreSchemaError("Database runtime schema version is unsupported")

    async def create_session(
        self,
        namespace: str,
        *,
        pin: ExecutionPin,
    ) -> RuntimeSessionSnapshot:
        return await asyncio.to_thread(self._create_session, namespace, pin)

    async def list_sessions(self, namespace: str) -> list[str]:
        return await asyncio.to_thread(self._list_runtime_sessions, namespace)

    async def load_session(
        self,
        namespace: str,
        session_id: str,
    ) -> RuntimeSessionSnapshot:
        return await asyncio.to_thread(self._load_runtime_session, namespace, session_id)

    async def delete_session(
        self,
        namespace: str,
        session_id: str,
        *,
        expected_revision: str,
    ) -> None:
        await asyncio.to_thread(
            self._delete_runtime_session,
            namespace,
            session_id,
            expected_revision,
        )

    @asynccontextmanager
    async def lock_session(
        self,
        namespace: str,
        session_id: str,
        *,
        timeout: float,
    ) -> AsyncIterator[None]:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        if timeout <= 0:
            raise ValueError("Session lock timeout must be positive")
        self._locks_directory.mkdir(parents=True, exist_ok=True)
        lock_name = sha256(f"{validated_namespace}\0{validated_id}".encode()).hexdigest()
        lock = AsyncFileLock(self._locks_directory / f"{lock_name}.lock")
        try:
            await lock.acquire(timeout=timeout)
        except FileLockTimeout as error:
            raise SessionLockTimeoutError(
                f"Session {session_id!r} remained busy for {timeout:g} seconds"
            ) from error
        try:
            yield
        finally:
            await lock.release()

    async def replace_session(
        self,
        namespace: str,
        session_id: str,
        *,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> RuntimeSessionSnapshot:
        return await asyncio.to_thread(
            self._replace_runtime_session,
            namespace,
            session_id,
            expected_revision,
            messages,
        )

    def _create_session(self, namespace: str, pin: ExecutionPin) -> RuntimeSessionSnapshot:
        validated_namespace = validate_namespace(namespace)
        revision = _revision(_EMPTY_MESSAGES)
        while True:
            session_id = new_session_id()
            try:
                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO runtime_sessions (
                            namespace, id, agent_spec_id, model_id, revision, messages
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            validated_namespace,
                            session_id,
                            pin.agent_spec_id,
                            pin.model_id,
                            revision,
                            _EMPTY_MESSAGES,
                        ),
                    )
            except sqlite3.IntegrityError as error:
                if error.sqlite_errorcode not in {
                    sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY,
                    sqlite3.SQLITE_CONSTRAINT_UNIQUE,
                }:
                    raise
                continue
            return RuntimeSessionSnapshot(
                id=session_id,
                namespace=validated_namespace,
                pin=pin,
                messages=[],
                revision=revision,
            )

    def _list_runtime_sessions(self, namespace: str) -> list[str]:
        validated_namespace = validate_namespace(namespace)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM runtime_sessions WHERE namespace = ? ORDER BY id",
                (validated_namespace,),
            ).fetchall()
        return [row[0] for row in rows]

    def _load_runtime_session(
        self,
        namespace: str,
        session_id: str,
    ) -> RuntimeSessionSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT agent_spec_id, model_id, revision, messages
                FROM runtime_sessions
                WHERE namespace = ? AND id = ?
                """,
                (validated_namespace, validated_id),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session {session_id!r} does not exist")
        agent_spec_id, model_id, revision, content = row
        try:
            messages = ModelMessagesTypeAdapter.validate_json(content)
        except (ValueError, ValidationError) as error:
            raise SessionFormatError(f"Session {session_id!r} has invalid messages") from error
        return RuntimeSessionSnapshot(
            id=validated_id,
            namespace=validated_namespace,
            pin=ExecutionPin(agent_spec_id=agent_spec_id, model_id=model_id),
            messages=messages,
            revision=revision,
        )

    def _delete_runtime_session(
        self,
        namespace: str,
        session_id: str,
        expected_revision: str,
    ) -> None:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT revision FROM runtime_sessions
                WHERE namespace = ? AND id = ?
                """,
                (validated_namespace, validated_id),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session {session_id!r} does not exist")
            if row[0] != expected_revision:
                raise SessionConflictError(f"Session {session_id!r} has changed")
            connection.execute(
                "DELETE FROM runtime_sessions WHERE namespace = ? AND id = ?",
                (validated_namespace, validated_id),
            )

    def _replace_runtime_session(
        self,
        namespace: str,
        session_id: str,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> RuntimeSessionSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        content = ModelMessagesTypeAdapter.dump_json(list(messages))
        revision = _revision(content)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT agent_spec_id, model_id, revision
                FROM runtime_sessions
                WHERE namespace = ? AND id = ?
                """,
                (validated_namespace, validated_id),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session {session_id!r} does not exist")
            agent_spec_id, model_id, current_revision = row
            if current_revision != expected_revision:
                raise SessionConflictError(f"Session {session_id!r} has changed")
            connection.execute(
                """
                UPDATE runtime_sessions
                SET revision = ?, messages = ?
                WHERE namespace = ? AND id = ?
                """,
                (revision, content, validated_namespace, validated_id),
            )
        return RuntimeSessionSnapshot(
            id=validated_id,
            namespace=validated_namespace,
            pin=ExecutionPin(agent_spec_id=agent_spec_id, model_id=model_id),
            messages=list(messages),
            revision=revision,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, timeout=5)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection


def _revision(content: bytes) -> str:
    return sha256(content).hexdigest()


__all__ = ["SQLiteRuntimeStore", "SQLiteSessionStore"]
