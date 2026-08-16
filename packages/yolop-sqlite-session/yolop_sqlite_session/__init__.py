from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from filelock import AsyncFileLock
from filelock import Timeout as FileLockTimeout
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.usage import RunUsage
from pydantic_core import from_json, to_json
from yolop_session import (
    ExecutionPin,
    IdempotencyConflictError,
    RunAdmissionError,
    RunCompletion,
    RunNotFoundError,
    RunReservation,
    RunStateError,
    RunStatus,
    RuntimeRunSnapshot,
    RuntimeSessionSnapshot,
    RuntimeStoreSchemaError,
    SessionConflictError,
    SessionFormatError,
    SessionLockTimeoutError,
    SessionNotFoundError,
    SessionSnapshot,
    StoredRunEvent,
    new_session_id,
    validate_namespace,
    validate_session_id,
)

_EMPTY_MESSAGES = b"[]"
_USAGE_ADAPTER = TypeAdapter(RunUsage)
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
);
CREATE TABLE runtime_runs (
    namespace TEXT NOT NULL,
    id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    owner_id TEXT,
    lease_expires_at TEXT,
    output BLOB,
    usage BLOB,
    session_revision TEXT,
    error_code TEXT,
    error_detail TEXT,
    PRIMARY KEY (namespace, id),
    UNIQUE (namespace, session_id, idempotency_key),
    FOREIGN KEY (namespace, session_id)
        REFERENCES runtime_sessions (namespace, id) ON DELETE CASCADE
);
CREATE TABLE runtime_run_events (
    namespace TEXT NOT NULL,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (namespace, run_id, sequence),
    FOREIGN KEY (namespace, run_id)
        REFERENCES runtime_runs (namespace, id) ON DELETE CASCADE
)
"""
_RUN_COLUMNS = """
id, namespace, session_id, idempotency_key, prompt, status,
created_at, updated_at, owner_id, lease_expires_at, output, usage,
session_revision, error_code, error_detail
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
                connection.executescript(_RUNTIME_SCHEMA)
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

    async def reserve_run(
        self,
        namespace: str,
        session_id: str,
        *,
        idempotency_key: str,
        prompt: str,
        max_pending: int | None = None,
    ) -> RunReservation:
        return await asyncio.to_thread(
            self._reserve_run,
            namespace,
            session_id,
            idempotency_key,
            prompt,
            max_pending,
        )

    async def load_run(self, namespace: str, run_id: str) -> RuntimeRunSnapshot:
        return await asyncio.to_thread(self._load_run, namespace, run_id)

    async def claim_run(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot:
        return await asyncio.to_thread(
            self._claim_run,
            namespace,
            run_id,
            owner_id,
            lease_seconds,
        )

    async def renew_run_lease(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot:
        return await asyncio.to_thread(
            self._renew_run_lease,
            namespace,
            run_id,
            owner_id,
            lease_seconds,
        )

    async def append_run_event(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        event: str,
        data: str,
    ) -> StoredRunEvent:
        return await asyncio.to_thread(
            self._append_run_event,
            namespace,
            run_id,
            owner_id,
            event,
            data,
        )

    async def list_run_events(
        self,
        namespace: str,
        run_id: str,
        *,
        after: int = 0,
    ) -> list[StoredRunEvent]:
        return await asyncio.to_thread(self._list_run_events, namespace, run_id, after)

    async def complete_run(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        expected_session_revision: str,
        messages: Sequence[ModelMessage],
        output: Any,
        usage: RunUsage,
    ) -> RunCompletion:
        return await asyncio.to_thread(
            self._complete_run,
            namespace,
            run_id,
            owner_id,
            expected_session_revision,
            messages,
            output,
            usage,
        )

    async def fail_run(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        error_code: str,
        error_detail: str,
    ) -> RuntimeRunSnapshot:
        return await asyncio.to_thread(
            self._fail_run,
            namespace,
            run_id,
            owner_id,
            error_code,
            error_detail,
        )

    async def interrupt_owned_runs(self, owner_id: str) -> int:
        return await asyncio.to_thread(self._interrupt_owned_runs, owner_id)

    async def interrupt_expired_runs(self) -> int:
        return await asyncio.to_thread(self._interrupt_expired_runs)

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

    def _reserve_run(
        self,
        namespace: str,
        session_id: str,
        idempotency_key: str,
        prompt: str,
        max_pending: int | None,
    ) -> RunReservation:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        if not idempotency_key or len(idempotency_key) > 255:
            raise ValueError("Idempotency key must contain between 1 and 255 characters")
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session_exists = connection.execute(
                "SELECT 1 FROM runtime_sessions WHERE namespace = ? AND id = ?",
                (validated_namespace, validated_id),
            ).fetchone()
            if session_exists is None:
                raise SessionNotFoundError(f"Session {session_id!r} does not exist")
            row = connection.execute(
                f"""
                SELECT {_RUN_COLUMNS} FROM runtime_runs
                WHERE namespace = ? AND session_id = ? AND idempotency_key = ?
                """,
                (validated_namespace, validated_id, idempotency_key),
            ).fetchone()
            if row is not None:
                run = _runtime_run(row)
                if run.prompt != prompt:
                    raise IdempotencyConflictError(
                        f"Idempotency key {idempotency_key!r} has different input"
                    )
                return RunReservation(run=run, created=False)

            if max_pending is not None:
                if max_pending < 1:
                    raise ValueError("Maximum pending runs must be positive")
                pending_row = connection.execute(
                    """
                    SELECT COUNT(*) FROM runtime_runs
                    WHERE namespace = ? AND session_id = ? AND status IN (?, ?)
                    """,
                    (
                        validated_namespace,
                        validated_id,
                        RunStatus.ACCEPTED,
                        RunStatus.RUNNING,
                    ),
                ).fetchone()
                assert pending_row is not None
                if pending_row[0] >= max_pending:
                    raise RunAdmissionError(
                        f"Session {session_id!r} already has {max_pending} pending runs"
                    )

            run_id = new_session_id()
            connection.execute(
                """
                INSERT INTO runtime_runs (
                    namespace, id, session_id, idempotency_key, prompt, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validated_namespace,
                    run_id,
                    validated_id,
                    idempotency_key,
                    prompt,
                    RunStatus.ACCEPTED,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        return RunReservation(
            run=RuntimeRunSnapshot(
                id=run_id,
                namespace=validated_namespace,
                session_id=validated_id,
                idempotency_key=idempotency_key,
                prompt=prompt,
                status=RunStatus.ACCEPTED,
                created_at=now,
                updated_at=now,
            ),
            created=True,
        )

    def _load_run(self, namespace: str, run_id: str) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        with self._connect() as connection:
            return _select_runtime_run(connection, validated_namespace, validated_id)

    def _claim_run(
        self,
        namespace: str,
        run_id: str,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        _validate_lease(owner_id, lease_seconds)
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = _select_runtime_run(connection, validated_namespace, validated_id)
            if run.status is not RunStatus.ACCEPTED:
                raise RunStateError(f"Run {run_id!r} is not available to claim")
            connection.execute(
                """
                UPDATE runtime_runs
                SET status = ?, owner_id = ?, lease_expires_at = ?, updated_at = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    RunStatus.RUNNING,
                    owner_id,
                    lease_expires_at.isoformat(),
                    now.isoformat(),
                    validated_namespace,
                    validated_id,
                ),
            )
        return replace(
            run,
            status=RunStatus.RUNNING,
            updated_at=now,
            owner_id=owner_id,
            lease_expires_at=lease_expires_at,
        )

    def _renew_run_lease(
        self,
        namespace: str,
        run_id: str,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        _validate_lease(owner_id, lease_seconds)
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = _select_owned_running_run(
                connection,
                validated_namespace,
                validated_id,
                owner_id,
            )
            connection.execute(
                """
                UPDATE runtime_runs
                SET lease_expires_at = ?, updated_at = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    lease_expires_at.isoformat(),
                    now.isoformat(),
                    validated_namespace,
                    validated_id,
                ),
            )
        return replace(
            run,
            updated_at=now,
            lease_expires_at=lease_expires_at,
        )

    def _append_run_event(
        self,
        namespace: str,
        run_id: str,
        owner_id: str,
        event: str,
        data: str,
    ) -> StoredRunEvent:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _select_owned_running_run(
                connection,
                validated_namespace,
                validated_id,
                owner_id,
            )
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1
                FROM runtime_run_events
                WHERE namespace = ? AND run_id = ?
                """,
                (validated_namespace, validated_id),
            ).fetchone()
            assert row is not None
            sequence = row[0]
            connection.execute(
                """
                INSERT INTO runtime_run_events (namespace, run_id, sequence, event, data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (validated_namespace, validated_id, sequence, event, data),
            )
            connection.execute(
                """
                UPDATE runtime_runs SET updated_at = ?
                WHERE namespace = ? AND id = ?
                """,
                (now.isoformat(), validated_namespace, validated_id),
            )
        return StoredRunEvent(sequence=sequence, event=event, data=data)

    def _list_run_events(
        self,
        namespace: str,
        run_id: str,
        after: int,
    ) -> list[StoredRunEvent]:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        if after < 0:
            raise ValueError("Run event sequence cannot be negative")
        with self._connect() as connection:
            _select_runtime_run(connection, validated_namespace, validated_id)
            rows = connection.execute(
                """
                SELECT sequence, event, data FROM runtime_run_events
                WHERE namespace = ? AND run_id = ? AND sequence > ?
                ORDER BY sequence
                """,
                (validated_namespace, validated_id, after),
            ).fetchall()
        return [StoredRunEvent(sequence=row[0], event=row[1], data=row[2]) for row in rows]

    def _complete_run(
        self,
        namespace: str,
        run_id: str,
        owner_id: str,
        expected_session_revision: str,
        messages: Sequence[ModelMessage],
        output: Any,
        usage: RunUsage,
    ) -> RunCompletion:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        message_content = ModelMessagesTypeAdapter.dump_json(list(messages))
        session_revision = _revision(message_content)
        output_content = to_json(output)
        usage_content = _USAGE_ADAPTER.dump_json(usage)
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = _select_owned_running_run(
                connection,
                validated_namespace,
                validated_id,
                owner_id,
            )
            session_row = connection.execute(
                """
                SELECT agent_spec_id, model_id, revision
                FROM runtime_sessions
                WHERE namespace = ? AND id = ?
                """,
                (validated_namespace, run.session_id),
            ).fetchone()
            if session_row is None:
                raise SessionNotFoundError(f"Session {run.session_id!r} does not exist")
            agent_spec_id, model_id, current_revision = session_row
            if current_revision != expected_session_revision:
                raise SessionConflictError(f"Session {run.session_id!r} has changed")
            connection.execute(
                """
                UPDATE runtime_sessions SET revision = ?, messages = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    session_revision,
                    message_content,
                    validated_namespace,
                    run.session_id,
                ),
            )
            connection.execute(
                """
                UPDATE runtime_runs
                SET status = ?, updated_at = ?, owner_id = NULL, lease_expires_at = NULL,
                    output = ?, usage = ?, session_revision = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    RunStatus.COMPLETED,
                    now.isoformat(),
                    output_content,
                    usage_content,
                    session_revision,
                    validated_namespace,
                    validated_id,
                ),
            )
        session = RuntimeSessionSnapshot(
            id=run.session_id,
            namespace=validated_namespace,
            pin=ExecutionPin(agent_spec_id=agent_spec_id, model_id=model_id),
            messages=list(messages),
            revision=session_revision,
        )
        completed_run = replace(
            run,
            status=RunStatus.COMPLETED,
            updated_at=now,
            owner_id=None,
            lease_expires_at=None,
            output=output,
            usage=usage,
            session_revision=session_revision,
        )
        return RunCompletion(session=session, run=completed_run)

    def _fail_run(
        self,
        namespace: str,
        run_id: str,
        owner_id: str,
        error_code: str,
        error_detail: str,
    ) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        if not error_code or not error_detail:
            raise ValueError("Run failure requires a stable code and safe detail")
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = _select_owned_running_run(
                connection,
                validated_namespace,
                validated_id,
                owner_id,
            )
            connection.execute(
                """
                UPDATE runtime_runs
                SET status = ?, updated_at = ?, owner_id = NULL, lease_expires_at = NULL,
                    error_code = ?, error_detail = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    RunStatus.FAILED,
                    now.isoformat(),
                    error_code,
                    error_detail,
                    validated_namespace,
                    validated_id,
                ),
            )
        return replace(
            run,
            status=RunStatus.FAILED,
            updated_at=now,
            owner_id=None,
            lease_expires_at=None,
            error_code=error_code,
            error_detail=error_detail,
        )

    def _interrupt_owned_runs(self, owner_id: str) -> int:
        if not owner_id:
            raise ValueError("Run owner ID cannot be empty")
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runtime_runs
                SET status = ?, updated_at = ?, owner_id = NULL, lease_expires_at = NULL,
                    error_code = ?, error_detail = ?
                WHERE status = ? AND owner_id = ?
                """,
                (
                    RunStatus.INTERRUPTED,
                    now,
                    "run_interrupted",
                    "Run worker stopped before completion",
                    RunStatus.RUNNING,
                    owner_id,
                ),
            )
            return cursor.rowcount

    def _interrupt_expired_runs(self) -> int:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runtime_runs
                SET status = ?, updated_at = ?, owner_id = NULL, lease_expires_at = NULL,
                    error_code = ?, error_detail = ?
                WHERE status = ? AND lease_expires_at <= ?
                """,
                (
                    RunStatus.INTERRUPTED,
                    now,
                    "run_interrupted",
                    "Run worker stopped before completion",
                    RunStatus.RUNNING,
                    now,
                ),
            )
            return cursor.rowcount

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
        connection.execute("PRAGMA foreign_keys=ON")
        return connection


def _select_runtime_run(
    connection: sqlite3.Connection,
    namespace: str,
    run_id: str,
) -> RuntimeRunSnapshot:
    row = connection.execute(
        f"SELECT {_RUN_COLUMNS} FROM runtime_runs WHERE namespace = ? AND id = ?",
        (namespace, run_id),
    ).fetchone()
    if row is None:
        raise RunNotFoundError(f"Run {run_id!r} does not exist")
    return _runtime_run(row)


def _select_owned_running_run(
    connection: sqlite3.Connection,
    namespace: str,
    run_id: str,
    owner_id: str,
) -> RuntimeRunSnapshot:
    run = _select_runtime_run(connection, namespace, run_id)
    if run.status is not RunStatus.RUNNING or run.owner_id != owner_id:
        raise RunStateError(f"Run {run_id!r} is not running for this owner")
    return run


def _validate_lease(owner_id: str, lease_seconds: float) -> None:
    if not owner_id:
        raise ValueError("Run owner ID cannot be empty")
    if lease_seconds <= 0:
        raise ValueError("Run lease must be positive")


def _runtime_run(row: sqlite3.Row | tuple[Any, ...]) -> RuntimeRunSnapshot:
    (
        run_id,
        namespace,
        session_id,
        idempotency_key,
        prompt,
        status,
        created_at,
        updated_at,
        owner_id,
        lease_expires_at,
        output,
        usage,
        session_revision,
        error_code,
        error_detail,
    ) = row
    return RuntimeRunSnapshot(
        id=run_id,
        namespace=namespace,
        session_id=session_id,
        idempotency_key=idempotency_key,
        prompt=prompt,
        status=RunStatus(status),
        created_at=datetime.fromisoformat(created_at),
        updated_at=datetime.fromisoformat(updated_at),
        owner_id=owner_id,
        lease_expires_at=(
            datetime.fromisoformat(lease_expires_at) if lease_expires_at is not None else None
        ),
        output=from_json(output) if output is not None else None,
        usage=_USAGE_ADAPTER.validate_json(usage) if usage is not None else None,
        session_revision=session_revision,
        error_code=error_code,
        error_detail=error_detail,
    )


def _revision(content: bytes) -> str:
    return sha256(content).hexdigest()


__all__ = ["SQLiteRuntimeStore", "SQLiteSessionStore"]
