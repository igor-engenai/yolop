from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from psycopg import AsyncConnection, sql
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.usage import RunUsage
from yolop_runtime import (
    IdempotencyConflictError,
    RootBudgetSnapshot,
    RunAdmissionError,
    RunBudgetExceededError,
    RunNotFoundError,
    RunRelation,
    RunReservation,
    RunStatus,
    RuntimeBudget,
    RuntimeRunSnapshot,
    input_digest,
    new_session_id,
    validate_namespace,
    validate_session_id,
)


class _PooledStore(Protocol):
    _pool: AsyncConnectionPool[AsyncConnection[Any]]


_USAGE_ADAPTER = TypeAdapter(RunUsage)
_RUN_SELECT = sql.SQL(
    """
    SELECT id, session_id, parent_run_id, root_run_id, relation, initiator,
           input_digest, full_message_start, full_message_end,
           active_message_start, active_message_end, idempotency_key, prompt,
           status, created_at, updated_at, owner_id, lease_expires_at,
           output, usage, session_revision, error_code, error_detail
    FROM yolop_runtime_runs
    """
)


def _as_uuid(value: str) -> UUID:
    return UUID(validate_session_id(value))


def _as_id(value: UUID | str | None) -> str | None:
    return None if value is None else str(value)


def _message_payload(messages: Sequence[ModelMessage]) -> list[Any]:
    return json.loads(ModelMessagesTypeAdapter.dump_json(list(messages)))


def _message_digest(messages: Sequence[ModelMessage]) -> bytes:
    return ModelMessagesTypeAdapter.dump_json(list(messages))


def _revision(messages: Sequence[ModelMessage]) -> str:
    return hashlib.sha256(_message_digest(messages)).hexdigest()


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _required_aware(value: datetime) -> datetime:
    aware = _aware(value)
    assert aware is not None
    return aware


def _validate_relation(
    relation: RunRelation | str,
    *,
    parent_run_id: str | None,
    root_run_id: str | None,
    root_budget: RuntimeBudget | None,
) -> RunRelation:
    try:
        validated = RunRelation(relation)
    except (TypeError, ValueError) as error:
        raise ValueError("Run relation is unsupported") from error
    if validated is RunRelation.ROOT and (parent_run_id is not None or root_run_id is not None):
        raise ValueError("A root Run cannot have an ancestor")
    if validated is not RunRelation.ROOT and parent_run_id is None:
        raise ValueError("A related Run requires a parent Run")
    if validated is not RunRelation.ROOT and root_budget is not None:
        raise ValueError("Only a root Run can define a root budget")
    return validated


def _message_range(start: int | None, end: int | None) -> tuple[int, int] | None:
    if start is None or end is None:
        return None
    return start, end


async def _append_message_range(
    connection: AsyncConnection[Any],
    *,
    namespace: str,
    session_id: UUID,
    messages: Sequence[ModelMessage],
) -> tuple[int, int] | None:
    payload = _message_payload(messages)
    if not payload:
        return None
    cursor = await connection.execute(
        """
        SELECT COALESCE(MAX(sequence), 0)
        FROM yolop_runtime_session_messages
        WHERE namespace = %s AND session_id = %s
        """,
        (namespace, session_id),
    )
    row = await cursor.fetchone()
    assert row is not None
    start = row[0] + 1
    for offset, message in enumerate(payload):
        await connection.execute(
            """
            INSERT INTO yolop_runtime_session_messages (
                namespace, session_id, sequence, message
            ) VALUES (%s, %s, %s, %s)
            """,
            (namespace, session_id, start + offset, Jsonb(message)),
        )
    return start, start + len(payload) - 1


async def _load_message_range(
    connection: AsyncConnection[Any],
    *,
    namespace: str,
    session_id: UUID,
    message_range: tuple[int, int] | None,
) -> list[ModelMessage]:
    if message_range is None:
        return []
    start, end = message_range
    cursor = await connection.execute(
        """
        SELECT message
        FROM yolop_runtime_session_messages
        WHERE namespace = %s AND session_id = %s
          AND sequence BETWEEN %s AND %s
        ORDER BY sequence
        """,
        (namespace, session_id, start, end),
    )
    payload = [row[0] for row in await cursor.fetchall()]
    return ModelMessagesTypeAdapter.validate_json(json.dumps(payload, separators=(",", ":")))


async def _run_snapshot(
    connection: AsyncConnection[Any],
    *,
    namespace: str,
    row: tuple[Any, ...],
) -> RuntimeRunSnapshot:
    (
        run_id,
        session_id,
        parent_run_id,
        root_run_id,
        relation,
        initiator,
        run_input_digest,
        full_message_start,
        full_message_end,
        active_message_start,
        active_message_end,
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
    session_uuid = session_id if isinstance(session_id, UUID) else UUID(str(session_id))
    full_messages = await _load_message_range(
        connection,
        namespace=namespace,
        session_id=session_uuid,
        message_range=_message_range(full_message_start, full_message_end),
    )
    active_messages = await _load_message_range(
        connection,
        namespace=namespace,
        session_id=session_uuid,
        message_range=_message_range(active_message_start, active_message_end),
    )
    return RuntimeRunSnapshot(
        id=str(run_id),
        namespace=namespace,
        session_id=str(session_id),
        idempotency_key=idempotency_key,
        prompt=prompt,
        status=RunStatus(status),
        created_at=_required_aware(created_at),
        updated_at=_required_aware(updated_at),
        owner_id=owner_id,
        lease_expires_at=_aware(lease_expires_at),
        output=output,
        usage=_USAGE_ADAPTER.validate_python(usage) if usage is not None else None,
        session_revision=session_revision,
        error_code=error_code,
        error_detail=error_detail,
        parent_run_id=_as_id(parent_run_id),
        root_run_id=_as_id(root_run_id),
        relation=RunRelation(relation),
        initiator=initiator,
        input_digest=run_input_digest,
        full_messages=full_messages,
        active_messages=active_messages,
        events=[],
    )


def _budget_from_row(
    namespace: str,
    root_run_id: str,
    row: tuple[Any, ...],
) -> RootBudgetSnapshot:
    (
        request_limit,
        input_tokens_limit,
        output_tokens_limit,
        total_tokens_limit,
        child_run_limit,
        continuation_limit,
        wall_deadline,
        requests_used,
        input_tokens_used,
        output_tokens_used,
        total_tokens_used,
        child_runs_used,
        continuations_used,
        active_runs,
        stopped,
        updated_at,
    ) = row
    return RootBudgetSnapshot(
        namespace=namespace,
        root_run_id=root_run_id,
        budget=RuntimeBudget(
            request_limit=request_limit,
            input_tokens_limit=input_tokens_limit,
            output_tokens_limit=output_tokens_limit,
            total_tokens_limit=total_tokens_limit,
            child_run_limit=child_run_limit,
            continuation_limit=continuation_limit,
            wall_deadline=_aware(wall_deadline),
        ),
        requests_used=requests_used,
        input_tokens_used=input_tokens_used,
        output_tokens_used=output_tokens_used,
        total_tokens_used=total_tokens_used,
        child_runs_used=child_runs_used,
        continuations_used=continuations_used,
        active_runs=active_runs,
        stopped=stopped,
        updated_at=_aware(updated_at),
    )


def _check_related_budget(
    row: tuple[Any, ...],
    *,
    relation: RunRelation,
    now: datetime,
) -> None:
    (
        request_limit,
        input_tokens_limit,
        output_tokens_limit,
        total_tokens_limit,
        child_run_limit,
        continuation_limit,
        wall_deadline,
        requests_used,
        input_tokens_used,
        output_tokens_used,
        total_tokens_used,
        child_runs_used,
        continuations_used,
        _active_runs,
        stopped,
    ) = row
    if stopped:
        raise RunBudgetExceededError("Root execution was cancelled")
    deadline = _aware(wall_deadline)
    if deadline is not None and deadline <= now:
        raise RunBudgetExceededError("Root wall deadline has passed")
    if request_limit is not None and requests_used >= request_limit:
        raise RunBudgetExceededError("Root request budget is exhausted")
    if input_tokens_limit is not None and input_tokens_used >= input_tokens_limit:
        raise RunBudgetExceededError("Root input-token budget is exhausted")
    if output_tokens_limit is not None and output_tokens_used >= output_tokens_limit:
        raise RunBudgetExceededError("Root output-token budget is exhausted")
    if total_tokens_limit is not None and total_tokens_used >= total_tokens_limit:
        raise RunBudgetExceededError("Root total-token budget is exhausted")
    if relation is RunRelation.CHILD:
        if child_run_limit is not None and child_runs_used >= child_run_limit:
            raise RunBudgetExceededError("Root child Run budget is exhausted")
    elif continuation_limit is not None and continuations_used >= continuation_limit:
        raise RunBudgetExceededError("Root continuation budget is exhausted")


async def _insert_root_budget(
    connection: AsyncConnection[Any],
    *,
    namespace: str,
    root_run_id: UUID,
    budget: RuntimeBudget,
    now: datetime,
) -> None:
    if budget.wall_deadline is not None and budget.wall_deadline <= now:
        raise RunBudgetExceededError("Root wall deadline has passed")
    if budget.request_limit == 0:
        raise RunBudgetExceededError("Root request budget is exhausted")
    await connection.execute(
        """
        INSERT INTO yolop_runtime_root_budgets (
            namespace, root_run_id, request_limit, input_tokens_limit,
            output_tokens_limit, total_tokens_limit, child_run_limit,
            continuation_limit, wall_deadline, active_runs
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1)
        """,
        (
            namespace,
            root_run_id,
            budget.request_limit,
            budget.input_tokens_limit,
            budget.output_tokens_limit,
            budget.total_tokens_limit,
            budget.child_run_limit,
            budget.continuation_limit,
            budget.wall_deadline,
        ),
    )


class PostgresRunOperations:
    """Private PostgreSQL Run reservation, history, and budget operations."""

    _pool: AsyncConnectionPool[AsyncConnection[Any]]

    async def reserve_run(
        self: _PooledStore,
        namespace: str,
        session_id: str,
        *,
        idempotency_key: str,
        prompt: str,
        max_pending: int | None = None,
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
        relation: RunRelation = RunRelation.ROOT,
        root_budget: RuntimeBudget | None = None,
        initiator: str = "user",
        input_digest: str | None = None,
        full_messages: Sequence[ModelMessage] = (),
        active_messages: Sequence[ModelMessage] = (),
    ) -> RunReservation:
        validated_namespace = validate_namespace(namespace)
        validated_session_id = validate_session_id(session_id)
        if not idempotency_key or len(idempotency_key) > 255:
            raise ValueError("Idempotency key must contain between 1 and 255 characters")
        if not initiator.strip():
            raise ValueError("Run initiator must not be empty")
        validated_parent_id = (
            validate_session_id(parent_run_id) if parent_run_id is not None else None
        )
        validated_root_id = validate_session_id(root_run_id) if root_run_id is not None else None
        validated_relation = _validate_relation(
            relation,
            parent_run_id=validated_parent_id,
            root_run_id=validated_root_id,
            root_budget=root_budget,
        )
        if max_pending is not None and max_pending < 1:
            raise ValueError("Maximum pending runs must be positive")
        run_input_digest = input_digest or input_digest_fn(prompt)
        session_uuid = _as_uuid(validated_session_id)
        now = datetime.now(UTC)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                session_cursor = await connection.execute(
                    """
                    SELECT 1
                    FROM yolop_runtime_sessions
                    WHERE namespace = %s AND id = %s
                    FOR UPDATE
                    """,
                    (validated_namespace, session_uuid),
                )
                if await session_cursor.fetchone() is None:
                    from yolop_runtime import SessionNotFoundError

                    raise SessionNotFoundError(
                        f"Session {validated_session_id!r} does not exist"
                    )
                existing_cursor = await connection.execute(
                    _RUN_SELECT
                    + sql.SQL(
                        "WHERE namespace = %s AND session_id = %s AND idempotency_key = %s"
                    ),
                    (validated_namespace, session_uuid, idempotency_key),
                )
                existing_row = await existing_cursor.fetchone()
                if existing_row is not None:
                    existing = await _run_snapshot(
                        connection,
                        namespace=validated_namespace,
                        row=existing_row,
                    )
                    if existing.prompt != prompt or existing.input_digest != run_input_digest:
                        raise IdempotencyConflictError(
                            f"Idempotency key {idempotency_key!r} has different input"
                        )
                    return RunReservation(run=existing, created=False)

                if max_pending is not None:
                    pending_cursor = await connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM yolop_runtime_runs
                        WHERE namespace = %s AND session_id = %s
                          AND status IN ('accepted', 'running')
                        """,
                        (validated_namespace, session_uuid),
                    )
                    pending_row = await pending_cursor.fetchone()
                    assert pending_row is not None
                    if pending_row[0] >= max_pending:
                        raise RunAdmissionError(
                            f"Session {validated_session_id!r} already has "
                            f"{max_pending} pending runs"
                        )

                run_id = new_session_id()
                run_uuid = _as_uuid(run_id)
                effective_root_id = run_uuid
                budget_row: tuple[Any, ...] | None = None
                if validated_relation is not RunRelation.ROOT:
                    assert validated_parent_id is not None
                    parent_cursor = await connection.execute(
                        """
                        SELECT session_id, root_run_id
                        FROM yolop_runtime_runs
                        WHERE namespace = %s AND id = %s
                        """,
                        (validated_namespace, _as_uuid(validated_parent_id)),
                    )
                    parent_row = await parent_cursor.fetchone()
                    if parent_row is None or parent_row[0] != session_uuid:
                        raise RunNotFoundError(
                            f"Parent Run {validated_parent_id!r} does not exist"
                        )
                    effective_root_id = parent_row[1]
                    if (
                        validated_root_id is not None
                        and validated_root_id != str(effective_root_id)
                    ):
                        raise ValueError("Related Run root does not match its parent")
                    budget_cursor = await connection.execute(
                        """
                        SELECT request_limit, input_tokens_limit, output_tokens_limit,
                               total_tokens_limit, child_run_limit, continuation_limit,
                               wall_deadline, requests_used, input_tokens_used,
                               output_tokens_used, total_tokens_used, child_runs_used,
                               continuations_used, active_runs, stopped
                        FROM yolop_runtime_root_budgets
                        WHERE namespace = %s AND root_run_id = %s
                        FOR UPDATE
                        """,
                        (validated_namespace, effective_root_id),
                    )
                    budget_row = await budget_cursor.fetchone()
                    if budget_row is not None:
                        _check_related_budget(
                            budget_row,
                            relation=validated_relation,
                            now=now,
                        )
                        counter = (
                            "child_runs_used"
                            if validated_relation is RunRelation.CHILD
                            else "continuations_used"
                        )
                        await connection.execute(
                            sql.SQL(
                                "UPDATE yolop_runtime_root_budgets "
                                "SET "
                                + counter
                                + " = "
                                + counter
                                + " + 1, active_runs = active_runs + 1, "
                                "updated_at = CURRENT_TIMESTAMP "
                                "WHERE namespace = %s AND root_run_id = %s"
                            ),
                            (validated_namespace, effective_root_id),
                        )

                full_range = await _append_message_range(
                    connection,
                    namespace=validated_namespace,
                    session_id=session_uuid,
                    messages=full_messages,
                )
                active_range = full_range
                if _message_digest(full_messages) != _message_digest(active_messages):
                    active_range = await _append_message_range(
                        connection,
                        namespace=validated_namespace,
                        session_id=session_uuid,
                        messages=active_messages,
                    )
                await connection.execute(
                    """
                    INSERT INTO yolop_runtime_runs (
                        namespace, id, session_id, parent_run_id, root_run_id,
                        relation, initiator, input_digest, full_message_start,
                        full_message_end, active_message_start, active_message_end,
                        idempotency_key, prompt, status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, 'accepted'
                    )
                    """,
                    (
                        validated_namespace,
                        run_uuid,
                        session_uuid,
                        _as_uuid(validated_parent_id) if validated_parent_id else None,
                        effective_root_id,
                        validated_relation.value,
                        initiator,
                        run_input_digest,
                        full_range[0] if full_range else None,
                        full_range[1] if full_range else None,
                        active_range[0] if active_range else None,
                        active_range[1] if active_range else None,
                        idempotency_key,
                        prompt,
                    ),
                )
                if validated_relation is RunRelation.ROOT and root_budget is not None:
                    await _insert_root_budget(
                        connection,
                        namespace=validated_namespace,
                        root_run_id=run_uuid,
                        budget=root_budget,
                        now=now,
                    )
                snapshot = await _run_snapshot(
                    connection,
                    namespace=validated_namespace,
                    row=(
                        run_uuid,
                        session_uuid,
                        _as_uuid(validated_parent_id) if validated_parent_id else None,
                        effective_root_id,
                        validated_relation.value,
                        initiator,
                        run_input_digest,
                        full_range[0] if full_range else None,
                        full_range[1] if full_range else None,
                        active_range[0] if active_range else None,
                        active_range[1] if active_range else None,
                        idempotency_key,
                        prompt,
                        RunStatus.ACCEPTED.value,
                        now,
                        now,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                )
                return RunReservation(run=snapshot, created=True)

    async def load_run(
        self: _PooledStore,
        namespace: str,
        run_id: str,
    ) -> RuntimeRunSnapshot:
        from yolop_runtime import RunNotFoundError

        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                _RUN_SELECT + sql.SQL("WHERE namespace = %s AND id = %s"),
                (validated_namespace, _as_uuid(validated_id)),
            )
            row = await cursor.fetchone()
            if row is None:
                raise RunNotFoundError(f"Run {run_id!r} does not exist")
            return await _run_snapshot(connection, namespace=validated_namespace, row=row)

    async def list_runs(
        self: _PooledStore,
        namespace: str,
        *,
        session_id: str | None = None,
    ) -> list[RuntimeRunSnapshot]:
        validated_namespace = validate_namespace(namespace)
        validated_session_id = validate_session_id(session_id) if session_id else None
        query = _RUN_SELECT + sql.SQL("WHERE namespace = %s")
        params: list[Any] = [validated_namespace]
        if validated_session_id is not None:
            query += sql.SQL(" AND session_id = %s")
            params.append(_as_uuid(validated_session_id))
        query += sql.SQL(" ORDER BY created_at, id")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(query, params)
            rows = await cursor.fetchall()
            return [
                await _run_snapshot(connection, namespace=validated_namespace, row=row)
                for row in rows
            ]

    async def load_root_budget(
        self: _PooledStore,
        namespace: str,
        root_run_id: str,
    ) -> RootBudgetSnapshot | None:
        validated_namespace = validate_namespace(namespace)
        validated_root_id = validate_session_id(root_run_id)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT request_limit, input_tokens_limit, output_tokens_limit,
                       total_tokens_limit, child_run_limit, continuation_limit,
                       wall_deadline, requests_used, input_tokens_used,
                       output_tokens_used, total_tokens_used, child_runs_used,
                       continuations_used, active_runs, stopped, updated_at
                FROM yolop_runtime_root_budgets
                WHERE namespace = %s AND root_run_id = %s
                """,
                (validated_namespace, _as_uuid(validated_root_id)),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _budget_from_row(validated_namespace, validated_root_id, row)


# Keep the helper name distinct from the optional keyword argument in reserve_run.
input_digest_fn = input_digest


__all__ = ["PostgresRunOperations"]
