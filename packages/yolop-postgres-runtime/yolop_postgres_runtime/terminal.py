from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from psycopg import sql
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage
from yolop_runtime import (
    ExecutionPin,
    RunCompletion,
    RunNotFoundError,
    RunStateError,
    RunStatus,
    RuntimeRunSnapshot,
    RuntimeSessionSnapshot,
    SessionConflictError,
    SessionNotFoundError,
    StoredRunEvent,
    validate_namespace,
    validate_session_id,
)

from .claims import _owned_running_row
from .runs import (
    _RUN_SELECT,
    _append_message_range,
    _as_uuid,
    _message_digest,
    _message_payload,
    _PooledStore,
    _revision,
    _run_snapshot,
)


class _InterruptibleStore(_PooledStore, Protocol):
    async def _interrupt_runs(
        self,
        *,
        owner_id: str | None,
        expired_before: datetime | None,
    ) -> int: ...


def _json_event_data(data: str) -> Any:
    try:
        return json.loads(data)
    except (TypeError, json.JSONDecodeError):
        return data


def _usage_payload(usage: RunUsage) -> dict[str, Any]:
    return json.loads(TypeAdapter(RunUsage).dump_json(usage))


class PostgresTerminalOperations:
    """Private PostgreSQL event and terminal Run transitions."""

    async def append_run_event(
        self: _PooledStore,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        event: str,
        data: str,
    ) -> StoredRunEvent:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        now = datetime.now(UTC)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    _RUN_SELECT + sql.SQL("WHERE namespace = %s AND id = %s FOR UPDATE"),
                    (validated_namespace, _as_uuid(validated_id)),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RunNotFoundError(f"Run {run_id!r} does not exist")
                if not _owned_running_row(row, owner_id, now):
                    raise RunStateError(f"Run {run_id!r} is not running for this owner")
                sequence_cursor = await connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM yolop_runtime_run_events
                    WHERE namespace = %s AND run_id = %s
                    """,
                    (validated_namespace, _as_uuid(validated_id)),
                )
                sequence_row = await sequence_cursor.fetchone()
                assert sequence_row is not None
                sequence = sequence_row[0]
                await connection.execute(
                    """
                    INSERT INTO yolop_runtime_run_events (
                        namespace, run_id, sequence, event, data
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        validated_namespace,
                        _as_uuid(validated_id),
                        sequence,
                        event,
                        Jsonb(_json_event_data(data)),
                    ),
                )
                await connection.execute(
                    """
                    UPDATE yolop_runtime_runs
                    SET updated_at = %s
                    WHERE namespace = %s AND id = %s
                    """,
                    (now, validated_namespace, _as_uuid(validated_id)),
                )
                if getattr(self, "_notifications_enabled", True):
                    await connection.execute(
                        "SELECT pg_notify(%s, %s)",
                        (
                            "yolop_runtime_events",
                            f"{validated_namespace}:{validated_id}:{sequence}",
                        ),
                    )
        return StoredRunEvent(sequence=sequence, event=event, data=data)

    async def list_run_events(
        self: _PooledStore,
        namespace: str,
        run_id: str,
        *,
        after: int = 0,
    ) -> list[StoredRunEvent]:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        if after < 0:
            raise ValueError("Run event sequence cannot be negative")
        async with self._pool.connection() as connection:
            run_cursor = await connection.execute(
                """
                SELECT 1
                FROM yolop_runtime_runs
                WHERE namespace = %s AND id = %s
                """,
                (validated_namespace, _as_uuid(validated_id)),
            )
            if await run_cursor.fetchone() is None:
                raise RunNotFoundError(f"Run {run_id!r} does not exist")
            cursor = await connection.execute(
                """
                SELECT sequence, event, data
                FROM yolop_runtime_run_events
                WHERE namespace = %s AND run_id = %s AND sequence > %s
                ORDER BY sequence
                """,
                (validated_namespace, _as_uuid(validated_id), after),
            )
            rows = await cursor.fetchall()
        return [
            StoredRunEvent(
                sequence=row[0],
                event=row[1],
                data=json.dumps(row[2], separators=(",", ":")),
            )
            for row in rows
        ]

    async def complete_run(
        self: _PooledStore,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        expected_session_revision: str,
        messages: Sequence[ModelMessage] | None = None,
        full_messages: Sequence[ModelMessage] | None = None,
        active_messages: Sequence[ModelMessage] | None = None,
        output: Any,
        usage: RunUsage,
    ) -> RunCompletion:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        full_list = list(full_messages if full_messages is not None else messages or ())
        active_list = list(active_messages if active_messages is not None else messages or ())
        revision = _revision(active_list)
        now = datetime.now(UTC)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    _RUN_SELECT + sql.SQL("WHERE namespace = %s AND id = %s FOR UPDATE"),
                    (validated_namespace, _as_uuid(validated_id)),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RunNotFoundError(f"Run {run_id!r} does not exist")
                if not _owned_running_row(row, owner_id, now):
                    raise RunStateError(f"Run {run_id!r} is not running for this owner")
                session_id = row[1]
                session_cursor = await connection.execute(
                    """
                    SELECT agent_spec_id, model_id, revision
                    FROM yolop_runtime_sessions
                    WHERE namespace = %s AND id = %s
                    FOR UPDATE
                    """,
                    (validated_namespace, session_id),
                )
                session_row = await session_cursor.fetchone()
                if session_row is None:
                    raise SessionNotFoundError(f"Session {session_id!r} does not exist")
                agent_spec_id, model_id, current_revision = session_row
                if current_revision != expected_session_revision:
                    raise SessionConflictError(f"Session {session_id!r} has changed")
                full_range = await _append_message_range(
                    connection,
                    namespace=validated_namespace,
                    session_id=session_id,
                    messages=full_list,
                )
                active_range = full_range
                if _message_digest(full_list) != _message_digest(active_list):
                    active_range = await _append_message_range(
                        connection,
                        namespace=validated_namespace,
                        session_id=session_id,
                        messages=active_list,
                    )
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
                        session_id,
                        revision,
                        Jsonb(_message_payload(active_list)),
                    ),
                )
                await connection.execute(
                    """
                    UPDATE yolop_runtime_sessions
                    SET revision = %s, head_run_id = %s, updated_at = %s
                    WHERE namespace = %s AND id = %s
                    """,
                    (
                        revision,
                        _as_uuid(validated_id),
                        now,
                        validated_namespace,
                        session_id,
                    ),
                )
                await connection.execute(
                    """
                    UPDATE yolop_runtime_runs
                    SET status = 'completed', updated_at = %s,
                        owner_id = NULL, lease_expires_at = NULL,
                        output = %s, usage = %s, session_revision = %s,
                        full_message_start = %s, full_message_end = %s,
                        active_message_start = %s, active_message_end = %s
                    WHERE namespace = %s AND id = %s
                    """,
                    (
                        now,
                        Jsonb(output),
                        Jsonb(_usage_payload(usage)),
                        revision,
                        full_range[0] if full_range else None,
                        full_range[1] if full_range else None,
                        active_range[0] if active_range else None,
                        active_range[1] if active_range else None,
                        validated_namespace,
                        _as_uuid(validated_id),
                    ),
                )
                await _account_budget(
                    connection,
                    namespace=validated_namespace,
                    root_run_id=row[3],
                    usage=usage,
                    stopped=False,
                    now=now,
                )
                updated = list(row)
                updated[7] = full_range[0] if full_range else None
                updated[8] = full_range[1] if full_range else None
                updated[9] = active_range[0] if active_range else None
                updated[10] = active_range[1] if active_range else None
                updated[13] = RunStatus.COMPLETED.value
                updated[15] = now
                updated[16] = None
                updated[17] = None
                updated[18] = output
                updated[19] = _usage_payload(usage)
                updated[20] = revision
                completed_run = await _run_snapshot(
                    connection,
                    namespace=validated_namespace,
                    row=tuple(updated),
                )
        completed_session = RuntimeSessionSnapshot(
            id=str(session_id),
            namespace=validated_namespace,
            pin=ExecutionPin(agent_spec_id=agent_spec_id, model_id=model_id),
            messages=active_list,
            revision=revision,
            head_run_id=validated_id,
        )
        return RunCompletion(session=completed_session, run=completed_run)

    async def fail_run(
        self: _PooledStore,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        error_code: str,
        error_detail: str,
    ) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        if not error_code or not error_detail:
            raise ValueError("Run failure requires a stable code and safe detail")
        now = datetime.now(UTC)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    _RUN_SELECT + sql.SQL("WHERE namespace = %s AND id = %s FOR UPDATE"),
                    (validated_namespace, _as_uuid(validated_id)),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RunNotFoundError(f"Run {run_id!r} does not exist")
                if not _owned_running_row(row, owner_id, now):
                    raise RunStateError(f"Run {run_id!r} is not running for this owner")
                await connection.execute(
                    """
                    UPDATE yolop_runtime_runs
                    SET status = 'failed', updated_at = %s,
                        owner_id = NULL, lease_expires_at = NULL,
                        error_code = %s, error_detail = %s
                    WHERE namespace = %s AND id = %s
                    """,
                    (
                        now,
                        error_code,
                        error_detail,
                        validated_namespace,
                        _as_uuid(validated_id),
                    ),
                )
                await _account_budget(
                    connection,
                    namespace=validated_namespace,
                    root_run_id=row[3],
                    usage=None,
                    stopped=False,
                    now=now,
                )
                updated = list(row)
                updated[13] = RunStatus.FAILED.value
                updated[15] = now
                updated[16] = None
                updated[17] = None
                updated[21] = error_code
                updated[22] = error_detail
                return await _run_snapshot(
                    connection,
                    namespace=validated_namespace,
                    row=tuple(updated),
                )

    async def cancel_run(
        self: _PooledStore,
        namespace: str,
        run_id: str,
        *,
        error_code: str = "run_cancelled",
        error_detail: str = "Run cancelled",
        expected_session_revision: str | None = None,
        full_messages: Sequence[ModelMessage] | None = None,
        active_messages: Sequence[ModelMessage] | None = None,
        output: Any | None = None,
        usage: RunUsage | None = None,
    ) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        if not error_code or not error_detail:
            raise ValueError("Run cancellation requires a stable code and safe detail")
        now = datetime.now(UTC)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    _RUN_SELECT + sql.SQL("WHERE namespace = %s AND id = %s FOR UPDATE"),
                    (validated_namespace, _as_uuid(validated_id)),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RunNotFoundError(f"Run {run_id!r} does not exist")
                if row[13] in {
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.INTERRUPTED.value,
                }:
                    return await _run_snapshot(
                        connection,
                        namespace=validated_namespace,
                        row=row,
                    )
                session_cursor = await connection.execute(
                    """
                    SELECT agent_spec_id, model_id, revision
                    FROM yolop_runtime_sessions
                    WHERE namespace = %s AND id = %s
                    FOR UPDATE
                    """,
                    (validated_namespace, row[1]),
                )
                session_row = await session_cursor.fetchone()
                if session_row is None:
                    raise SessionNotFoundError(f"Session {row[1]!r} does not exist")
                agent_spec_id, model_id, current_revision = session_row
                if (
                    expected_session_revision is not None
                    and current_revision != expected_session_revision
                ):
                    raise SessionConflictError(f"Session {row[1]!r} has changed")
                existing = await _run_snapshot(
                    connection,
                    namespace=validated_namespace,
                    row=row,
                )
                full_list = list(
                    full_messages if full_messages is not None else existing.full_messages
                )
                active_list = list(
                    active_messages if active_messages is not None else existing.active_messages
                )
                revision = _revision(active_list)
                full_range = await _append_message_range(
                    connection,
                    namespace=validated_namespace,
                    session_id=row[1],
                    messages=full_list,
                )
                active_range = full_range
                if _message_digest(full_list) != _message_digest(active_list):
                    active_range = await _append_message_range(
                        connection,
                        namespace=validated_namespace,
                        session_id=row[1],
                        messages=active_list,
                    )
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
                        row[1],
                        revision,
                        Jsonb(_message_payload(active_list)),
                    ),
                )
                await connection.execute(
                    """
                    UPDATE yolop_runtime_sessions
                    SET revision = %s, head_run_id = %s, updated_at = %s
                    WHERE namespace = %s AND id = %s
                    """,
                    (revision, _as_uuid(validated_id), now, validated_namespace, row[1]),
                )
                await connection.execute(
                    """
                    UPDATE yolop_runtime_runs
                    SET status = 'interrupted', updated_at = %s,
                        owner_id = NULL, lease_expires_at = NULL,
                        error_code = %s, error_detail = %s,
                        session_revision = %s, output = %s, usage = %s,
                        full_message_start = %s, full_message_end = %s,
                        active_message_start = %s, active_message_end = %s
                    WHERE namespace = %s AND id = %s
                    """,
                    (
                        now,
                        error_code,
                        error_detail,
                        revision,
                        Jsonb(output) if output is not None else None,
                        Jsonb(_usage_payload(usage)) if usage is not None else None,
                        full_range[0] if full_range else None,
                        full_range[1] if full_range else None,
                        active_range[0] if active_range else None,
                        active_range[1] if active_range else None,
                        validated_namespace,
                        _as_uuid(validated_id),
                    ),
                )
                await _account_budget(
                    connection,
                    namespace=validated_namespace,
                    root_run_id=row[3],
                    usage=usage,
                    stopped=row[4] == "root",
                    now=now,
                )
                updated = list(row)
                updated[7] = full_range[0] if full_range else None
                updated[8] = full_range[1] if full_range else None
                updated[9] = active_range[0] if active_range else None
                updated[10] = active_range[1] if active_range else None
                updated[13] = RunStatus.INTERRUPTED.value
                updated[15] = now
                updated[16] = None
                updated[17] = None
                updated[18] = output
                updated[19] = _usage_payload(usage) if usage is not None else None
                updated[20] = revision
                updated[21] = error_code
                updated[22] = error_detail
                return await _run_snapshot(
                    connection,
                    namespace=validated_namespace,
                    row=tuple(updated),
                )

    async def interrupt_owned_runs(self: _InterruptibleStore, owner_id: str) -> int:
        if not owner_id:
            raise ValueError("Run owner ID cannot be empty")
        return await self._interrupt_runs(owner_id=owner_id, expired_before=None)

    async def interrupt_expired_runs(self: _InterruptibleStore) -> int:
        return await self._interrupt_runs(owner_id=None, expired_before=datetime.now(UTC))

    async def _interrupt_runs(
        self: _InterruptibleStore,
        *,
        owner_id: str | None,
        expired_before: datetime | None,
    ) -> int:
        now = datetime.now(UTC)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                if owner_id is not None:
                    cursor = await connection.execute(
                        _RUN_SELECT
                        + sql.SQL("WHERE status = 'running' AND owner_id = %s FOR UPDATE"),
                        (owner_id,),
                    )
                else:
                    cursor = await connection.execute(
                        _RUN_SELECT
                        + sql.SQL(
                            "WHERE status = 'running' AND lease_expires_at <= %s FOR UPDATE"
                        ),
                        (expired_before,),
                    )
                rows = await cursor.fetchall()
                for row in rows:
                    namespace_cursor = await connection.execute(
                        "SELECT namespace FROM yolop_runtime_runs WHERE id = %s",
                        (row[0],),
                    )
                    namespace_row = await namespace_cursor.fetchone()
                    assert namespace_row is not None
                    run_namespace = namespace_row[0]
                    await connection.execute(
                        """
                        UPDATE yolop_runtime_runs
                        SET status = 'interrupted', updated_at = %s,
                            owner_id = NULL, lease_expires_at = NULL,
                            error_code = 'run_interrupted',
                            error_detail = 'Run worker stopped before completion'
                        WHERE namespace = %s AND id = %s
                        """,
                        (now, run_namespace, row[0]),
                    )
                    await _account_budget(
                        connection,
                        namespace=run_namespace,
                        root_run_id=row[3],
                        usage=None,
                        stopped=False,
                        now=now,
                    )
                return len(rows)


async def _account_budget(
    connection: Any,
    *,
    namespace: str,
    root_run_id: Any,
    usage: RunUsage | None,
    stopped: bool,
    now: datetime,
) -> None:
    if root_run_id is None or not namespace:
        return
    usage = usage or RunUsage()
    await connection.execute(
        """
        UPDATE yolop_runtime_root_budgets
        SET requests_used = requests_used + %s,
            input_tokens_used = input_tokens_used + %s,
            output_tokens_used = output_tokens_used + %s,
            total_tokens_used = total_tokens_used + %s,
            active_runs = GREATEST(active_runs - 1, 0),
            stopped = CASE WHEN %s THEN TRUE ELSE stopped END,
            updated_at = %s
        WHERE namespace = %s AND root_run_id = %s
        """,
        (
            usage.requests,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            stopped,
            now,
            namespace,
            root_run_id,
        ),
    )


__all__ = ["PostgresTerminalOperations"]
