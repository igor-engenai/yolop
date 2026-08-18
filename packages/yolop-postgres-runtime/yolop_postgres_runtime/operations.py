from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from psycopg import AsyncConnection, sql
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from yolop_runtime import (
    ExecutionPin,
    PluginStateEntry,
    RunNotFoundError,
    RunStateError,
    RunStatus,
    RuntimeSessionSnapshot,
    SessionConflictError,
    SessionFormatError,
    SessionLockTimeoutError,
    SessionNotFoundError,
    StateFormatError,
    StateSchemaError,
    StateScope,
    StateSequenceConflictError,
    encode_state_payload,
    new_session_id,
    validate_namespace,
    validate_session_id,
)


class _PooledStore(Protocol):
    _pool: AsyncConnectionPool[AsyncConnection[Any]]


def _message_payload(messages: Sequence[ModelMessage]) -> list[Any]:
    encoded = ModelMessagesTypeAdapter.dump_json(list(messages))
    return json.loads(encoded)


def _messages_from_payload(payload: Any, session_id: str) -> list[ModelMessage]:
    try:
        return ModelMessagesTypeAdapter.validate_json(json.dumps(payload, separators=(",", ":")))
    except (TypeError, ValueError) as error:
        raise SessionFormatError(f"Session {session_id!r} has invalid messages") from error


def _revision(messages: Sequence[ModelMessage]) -> str:
    return hashlib.sha256(ModelMessagesTypeAdapter.dump_json(list(messages))).hexdigest()


def _as_uuid(value: str) -> UUID:
    return UUID(validate_session_id(value))


def _as_id(value: UUID | str | None) -> str | None:
    return None if value is None else str(value)


def _state_text(value: str, field_name: str, limit: int = 128) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"State {field_name} must not be empty")
    if len(value) > limit:
        raise ValueError(f"State {field_name} exceeds {limit} characters")
    return value


def _state_scope(value: StateScope | str) -> StateScope:
    try:
        return StateScope(value)
    except (TypeError, ValueError) as error:
        raise StateFormatError("State scope is unsupported") from error


def _state_created_at(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _stream_key(
    namespace: str,
    owner_id: str,
    scope: StateScope,
    scope_id: str,
    state_kind: str,
) -> str:
    return ":".join((namespace, owner_id, scope.value, scope_id, state_kind))


class PostgresSessionOperations:
    """Private PostgreSQL Session and lock operations."""

    _pool: AsyncConnectionPool[AsyncConnection[Any]]

    async def create_session(
        self: _PooledStore,
        namespace: str,
        *,
        pin: ExecutionPin,
    ) -> RuntimeSessionSnapshot:
        validated_namespace = validate_namespace(namespace)
        session_id = new_session_id()
        messages: list[ModelMessage] = []
        revision = _revision(messages)
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO yolop_runtime_sessions (
                    namespace, id, agent_spec_id, model_id, revision
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    validated_namespace,
                    _as_uuid(session_id),
                    pin.agent_spec_id,
                    pin.model_id,
                    revision,
                ),
            )
            await connection.execute(
                """
                INSERT INTO yolop_runtime_session_contexts (
                    namespace, session_id, revision, messages
                ) VALUES (%s, %s, %s, %s)
                """,
                (validated_namespace, _as_uuid(session_id), revision, Jsonb(messages)),
            )
        return RuntimeSessionSnapshot(
            id=session_id,
            namespace=validated_namespace,
            pin=pin,
            messages=messages,
            revision=revision,
            head_run_id=None,
        )

    async def list_sessions(self: _PooledStore, namespace: str) -> list[str]:
        validated_namespace = validate_namespace(namespace)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT id
                FROM yolop_runtime_sessions
                WHERE namespace = %s
                ORDER BY id
                """,
                (validated_namespace,),
            )
            rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

    async def load_session(
        self: _PooledStore,
        namespace: str,
        session_id: str,
    ) -> RuntimeSessionSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT agent_spec_id, model_id, revision, head_run_id
                FROM yolop_runtime_sessions
                WHERE namespace = %s AND id = %s
                """,
                (validated_namespace, _as_uuid(validated_id)),
            )
            row = await cursor.fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session {session_id!r} does not exist")
            agent_spec_id, model_id, revision, head_run_id = row
            context_cursor = await connection.execute(
                """
                SELECT messages
                FROM yolop_runtime_session_contexts
                WHERE namespace = %s AND session_id = %s AND revision = %s
                """,
                (validated_namespace, _as_uuid(validated_id), revision),
            )
            context_row = await context_cursor.fetchone()
        if context_row is None:
            raise SessionFormatError(f"Session {session_id!r} has no active context")
        messages = _messages_from_payload(context_row[0], validated_id)
        return RuntimeSessionSnapshot(
            id=validated_id,
            namespace=validated_namespace,
            pin=ExecutionPin(agent_spec_id=agent_spec_id, model_id=model_id),
            messages=messages,
            revision=revision,
            head_run_id=_as_id(head_run_id),
        )

    async def checkout_session(
        self: _PooledStore,
        namespace: str,
        session_id: str,
        run_id: str,
        *,
        expected_revision: str,
    ) -> RuntimeSessionSnapshot:
        from .runs import _RUN_SELECT, _run_snapshot

        validated_namespace = validate_namespace(namespace)
        validated_session_id = validate_session_id(session_id)
        validated_run_id = validate_session_id(run_id)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                run_cursor = await connection.execute(
                    _RUN_SELECT + sql.SQL("WHERE namespace = %s AND id = %s"),
                    (validated_namespace, _as_uuid(validated_run_id)),
                )
                run_row = await run_cursor.fetchone()
                if run_row is None:
                    raise RunNotFoundError(f"Run {run_id!r} does not exist")
                run = await _run_snapshot(
                    connection,
                    namespace=validated_namespace,
                    row=run_row,
                )
                if run.session_id != validated_session_id:
                    raise RunStateError(f"Run {run_id!r} belongs to another session")
                if run.status not in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.INTERRUPTED,
                }:
                    raise RunStateError(f"Run {run_id!r} is not terminal")
                session_cursor = await connection.execute(
                    """
                    SELECT agent_spec_id, model_id, revision
                    FROM yolop_runtime_sessions
                    WHERE namespace = %s AND id = %s
                    FOR UPDATE
                    """,
                    (validated_namespace, _as_uuid(validated_session_id)),
                )
                session_row = await session_cursor.fetchone()
                if session_row is None:
                    raise SessionNotFoundError(f"Session {session_id!r} does not exist")
                agent_spec_id, model_id, current_revision = session_row
                if current_revision != expected_revision:
                    raise SessionConflictError(f"Session {session_id!r} has changed")
                revision = _revision(run.active_messages)
                await connection.execute(
                    """
                    INSERT INTO yolop_runtime_session_contexts (
                        namespace, session_id, revision, messages
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (namespace, session_id, revision)
                    DO UPDATE SET messages = EXCLUDED.messages
                    """,
                    (
                        validated_namespace,
                        _as_uuid(validated_session_id),
                        revision,
                        Jsonb(_message_payload(run.active_messages)),
                    ),
                )
                await connection.execute(
                    """
                    UPDATE yolop_runtime_sessions
                    SET revision = %s, head_run_id = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE namespace = %s AND id = %s
                    """,
                    (
                        revision,
                        _as_uuid(validated_run_id),
                        validated_namespace,
                        _as_uuid(validated_session_id),
                    ),
                )
        return RuntimeSessionSnapshot(
            id=validated_session_id,
            namespace=validated_namespace,
            pin=ExecutionPin(agent_spec_id=agent_spec_id, model_id=model_id),
            messages=list(run.active_messages),
            revision=revision,
            head_run_id=validated_run_id,
        )

    async def replace_session(
        self: _PooledStore,
        namespace: str,
        session_id: str,
        *,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> RuntimeSessionSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        message_list = list(messages)
        payload = _message_payload(message_list)
        revision = _revision(message_list)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    """
                    SELECT agent_spec_id, model_id, revision, head_run_id
                    FROM yolop_runtime_sessions
                    WHERE namespace = %s AND id = %s
                    FOR UPDATE
                    """,
                    (validated_namespace, _as_uuid(validated_id)),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise SessionNotFoundError(f"Session {session_id!r} does not exist")
                agent_spec_id, model_id, current_revision, head_run_id = row
                if current_revision != expected_revision:
                    raise SessionConflictError(f"Session {session_id!r} has changed")

                sequence_cursor = await connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0)
                    FROM yolop_runtime_session_messages
                    WHERE namespace = %s AND session_id = %s
                    """,
                    (validated_namespace, _as_uuid(validated_id)),
                )
                sequence_row = await sequence_cursor.fetchone()
                assert sequence_row is not None
                sequence = sequence_row[0]
                for message in payload:
                    sequence += 1
                    await connection.execute(
                        """
                        INSERT INTO yolop_runtime_session_messages (
                            namespace, session_id, sequence, message
                        ) VALUES (%s, %s, %s, %s)
                        """,
                        (
                            validated_namespace,
                            _as_uuid(validated_id),
                            sequence,
                            Jsonb(message),
                        ),
                    )
                await connection.execute(
                    """
                    INSERT INTO yolop_runtime_session_contexts (
                        namespace, session_id, revision, messages
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (namespace, session_id, revision)
                    DO UPDATE SET messages = EXCLUDED.messages
                    """,
                    (validated_namespace, _as_uuid(validated_id), revision, Jsonb(payload)),
                )
                await connection.execute(
                    """
                    UPDATE yolop_runtime_sessions
                    SET revision = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE namespace = %s AND id = %s
                    """,
                    (revision, validated_namespace, _as_uuid(validated_id)),
                )
        return RuntimeSessionSnapshot(
            id=validated_id,
            namespace=validated_namespace,
            pin=ExecutionPin(agent_spec_id=agent_spec_id, model_id=model_id),
            messages=message_list,
            revision=revision,
            head_run_id=_as_id(head_run_id),
        )

    async def delete_session(
        self: _PooledStore,
        namespace: str,
        session_id: str,
        *,
        expected_revision: str,
    ) -> None:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    """
                    SELECT revision
                    FROM yolop_runtime_sessions
                    WHERE namespace = %s AND id = %s
                    FOR UPDATE
                    """,
                    (validated_namespace, _as_uuid(validated_id)),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise SessionNotFoundError(f"Session {session_id!r} does not exist")
                if row[0] != expected_revision:
                    raise SessionConflictError(f"Session {session_id!r} has changed")
                await connection.execute(
                    """
                    DELETE FROM yolop_runtime_plugin_state
                    WHERE namespace = %s AND (
                        (scope = 'session' AND scope_id = %s)
                        OR (
                            scope = 'run' AND scope_id IN (
                                SELECT id
                                FROM yolop_runtime_runs
                                WHERE namespace = %s AND session_id = %s
                            )
                        )
                    )
                    """,
                    (
                        validated_namespace,
                        _as_uuid(validated_id),
                        validated_namespace,
                        _as_uuid(validated_id),
                    ),
                )
                await connection.execute(
                    """
                    DELETE FROM yolop_runtime_sessions
                    WHERE namespace = %s AND id = %s
                    """,
                    (validated_namespace, _as_uuid(validated_id)),
                )

    @asynccontextmanager
    async def lock_session(
        self: _PooledStore,
        namespace: str,
        session_id: str,
        *,
        timeout: float,
    ) -> AsyncIterator[None]:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        if timeout <= 0:
            raise ValueError("Session lock timeout must be positive")
        lock_key = f"{validated_namespace}:{validated_id}"
        deadline = time.monotonic() + timeout
        async with self._pool.connection() as connection:
            acquired = False
            while not acquired:
                cursor = await connection.execute(
                    "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
                    (lock_key,),
                )
                row = await cursor.fetchone()
                acquired = bool(row and row[0])
                await connection.commit()
                if acquired:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SessionLockTimeoutError(
                        f"Session {session_id!r} remained busy for {timeout:g} seconds"
                    )
                await asyncio.sleep(min(0.01, remaining))
            try:
                yield
            finally:
                await connection.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (lock_key,),
                )
                await connection.commit()


class PostgresStateOperations:
    """Private PostgreSQL scoped plugin-state operations."""

    _pool: AsyncConnectionPool[AsyncConnection[Any]]

    async def read_state(
        self: _PooledStore,
        namespace: str,
        *,
        owner_id: str,
        scope: StateScope,
        scope_id: str,
        state_kind: str,
        schema_version: int | None = None,
    ) -> list[PluginStateEntry]:
        validated_namespace = validate_namespace(namespace)
        validated_owner = _state_text(owner_id, "owner_id")
        validated_scope = _state_scope(scope)
        validated_scope_id = validate_session_id(scope_id)
        validated_kind = _state_text(state_kind, "state_kind")
        if schema_version is not None and (
            isinstance(schema_version, bool) or schema_version < 1
        ):
            raise StateSchemaError("State schema version must be positive")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT schema_version, sequence, payload, created_at
                FROM yolop_runtime_plugin_state
                WHERE namespace = %s AND owner_id = %s AND scope = %s
                  AND scope_id = %s AND state_kind = %s
                ORDER BY sequence
                """,
                (
                    validated_namespace,
                    validated_owner,
                    validated_scope.value,
                    _as_uuid(validated_scope_id),
                    validated_kind,
                ),
            )
            rows = await cursor.fetchall()
        entries = [
            PluginStateEntry(
                namespace=validated_namespace,
                owner_id=validated_owner,
                scope=validated_scope,
                scope_id=validated_scope_id,
                state_kind=validated_kind,
                schema_version=row[0],
                sequence=row[1],
                payload=row[2],
                created_at=_state_created_at(row[3]),
            )
            for row in rows
        ]
        if schema_version is not None and any(
            entry.schema_version != schema_version for entry in entries
        ):
            raise StateSchemaError(
                f"State stream {validated_kind!r} has an unsupported schema version"
            )
        return entries

    async def append_state(
        self: _PooledStore,
        namespace: str,
        *,
        owner_id: str,
        scope: StateScope,
        scope_id: str,
        state_kind: str,
        schema_version: int,
        expected_sequence: int,
        payload: Any,
    ) -> PluginStateEntry:
        validated_namespace = validate_namespace(namespace)
        validated_owner = _state_text(owner_id, "owner_id")
        validated_scope = _state_scope(scope)
        validated_scope_id = validate_session_id(scope_id)
        validated_kind = _state_text(state_kind, "state_kind")
        if isinstance(schema_version, bool) or schema_version < 1:
            raise StateSchemaError("State schema version must be positive")
        if isinstance(expected_sequence, bool) or expected_sequence < 0:
            raise ValueError("Expected state sequence cannot be negative")
        encoded = encode_state_payload(payload)
        stored_payload = json.loads(encoded)
        stream_key = _stream_key(
            validated_namespace,
            validated_owner,
            validated_scope,
            validated_scope_id,
            validated_kind,
        )
        async with self._pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (stream_key,),
                )
                cursor = await connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0)
                    FROM yolop_runtime_plugin_state
                    WHERE namespace = %s AND owner_id = %s AND scope = %s
                      AND scope_id = %s AND state_kind = %s
                    """,
                    (
                        validated_namespace,
                        validated_owner,
                        validated_scope.value,
                        _as_uuid(validated_scope_id),
                        validated_kind,
                    ),
                )
                row = await cursor.fetchone()
                assert row is not None
                current_sequence = row[0]
                if current_sequence != expected_sequence:
                    raise StateSequenceConflictError(
                        f"State stream {validated_kind!r} is at sequence {current_sequence}"
                    )
                sequence = current_sequence + 1
                cursor = await connection.execute(
                    """
                    INSERT INTO yolop_runtime_plugin_state (
                        namespace, owner_id, scope, scope_id, state_kind,
                        schema_version, sequence, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING created_at
                    """,
                    (
                        validated_namespace,
                        validated_owner,
                        validated_scope.value,
                        _as_uuid(validated_scope_id),
                        validated_kind,
                        schema_version,
                        sequence,
                        Jsonb(stored_payload),
                    ),
                )
                created_row = await cursor.fetchone()
                assert created_row is not None
        return PluginStateEntry(
            namespace=validated_namespace,
            owner_id=validated_owner,
            scope=validated_scope,
            scope_id=validated_scope_id,
            state_kind=validated_kind,
            schema_version=schema_version,
            sequence=sequence,
            payload=stored_payload,
            created_at=_state_created_at(created_row[0]),
        )


__all__ = ["PostgresSessionOperations", "PostgresStateOperations"]
