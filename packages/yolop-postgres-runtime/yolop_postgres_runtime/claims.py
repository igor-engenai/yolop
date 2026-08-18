from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg import AsyncConnection, sql
from psycopg_pool import AsyncConnectionPool
from yolop_runtime import (
    RunNotFoundError,
    RunStateError,
    RunStatus,
    RuntimeRunSnapshot,
    validate_namespace,
    validate_session_id,
)

from .runs import _RUN_SELECT, _as_uuid, _PooledStore, _run_snapshot


class PostgresClaimOperations:
    """Private PostgreSQL ownership and lease operations."""

    _pool: AsyncConnectionPool[AsyncConnection[Any]]

    async def claim_run(
        self: _PooledStore,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        _validate_lease(owner_id, lease_seconds)
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    _RUN_SELECT + sql.SQL("WHERE namespace = %s AND id = %s FOR UPDATE"),
                    (validated_namespace, _as_uuid(validated_id)),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise RunNotFoundError(f"Run {run_id!r} does not exist")
                if row[13] != RunStatus.ACCEPTED.value:
                    raise RunStateError(f"Run {run_id!r} is not available to claim")
                await connection.execute(
                    """
                    UPDATE yolop_runtime_runs
                    SET status = 'running', owner_id = %s,
                        lease_expires_at = %s, updated_at = %s
                    WHERE namespace = %s AND id = %s
                    """,
                    (
                        owner_id,
                        lease_expires_at,
                        now,
                        validated_namespace,
                        _as_uuid(validated_id),
                    ),
                )
                updated = list(row)
                updated[13] = RunStatus.RUNNING.value
                updated[15] = now
                updated[16] = owner_id
                updated[17] = lease_expires_at
                return await _run_snapshot(
                    connection,
                    namespace=validated_namespace,
                    row=tuple(updated),
                )

    async def renew_run_lease(
        self: _PooledStore,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        _validate_lease(owner_id, lease_seconds)
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
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
                    SET lease_expires_at = %s, updated_at = %s
                    WHERE namespace = %s AND id = %s
                    """,
                    (lease_expires_at, now, validated_namespace, _as_uuid(validated_id)),
                )
                updated = list(row)
                updated[15] = now
                updated[17] = lease_expires_at
                return await _run_snapshot(
                    connection,
                    namespace=validated_namespace,
                    row=tuple(updated),
                )


def _validate_lease(owner_id: str, lease_seconds: float) -> None:
    if not owner_id:
        raise ValueError("Run owner ID cannot be empty")
    if lease_seconds <= 0:
        raise ValueError("Run lease must be positive")


def _owned_running_row(row: Sequence[Any], owner_id: str, now: datetime) -> bool:
    lease_expires_at = row[17]
    if row[13] != RunStatus.RUNNING.value or row[16] != owner_id:
        return False
    return lease_expires_at is not None and lease_expires_at > now


__all__ = ["PostgresClaimOperations"]
