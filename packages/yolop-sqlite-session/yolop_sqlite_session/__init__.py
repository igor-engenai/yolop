from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
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
from yolop_runtime import (
    ExecutionPin,
    IdempotencyConflictError,
    PluginStateEntry,
    RootBudgetSnapshot,
    RunAdmissionError,
    RunBudgetExceededError,
    RunCompletion,
    RunNotFoundError,
    RunRelation,
    RunReservation,
    RunStateError,
    RunStatus,
    RuntimeBudget,
    RuntimeRunSnapshot,
    RuntimeSessionSnapshot,
    RuntimeStoreSchemaError,
    SessionConflictError,
    SessionFormatError,
    SessionLockTimeoutError,
    SessionNotFoundError,
    StateFormatError,
    StateSchemaError,
    StateScope,
    StateSequenceConflictError,
    StoredRunEvent,
    decode_state_payload,
    encode_state_payload,
    input_digest,
    new_session_id,
    validate_namespace,
    validate_session_id,
)

_EMPTY_MESSAGES = b"[]"
_USAGE_ADAPTER = TypeAdapter(RunUsage)
_RUNTIME_SCHEMA_VERSION = 4
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
    head_run_id TEXT,
    PRIMARY KEY (namespace, id)
);
CREATE TABLE runtime_runs (
    namespace TEXT NOT NULL,
    id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    parent_run_id TEXT,
    root_run_id TEXT,
    relation TEXT NOT NULL,
    initiator TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    full_messages BLOB NOT NULL,
    active_messages BLOB NOT NULL,
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
);
CREATE TABLE runtime_root_budgets (
    namespace TEXT NOT NULL,
    root_run_id TEXT NOT NULL,
    request_limit INTEGER,
    input_tokens_limit INTEGER,
    output_tokens_limit INTEGER,
    total_tokens_limit INTEGER,
    child_run_limit INTEGER,
    continuation_limit INTEGER,
    wall_deadline TEXT,
    requests_used INTEGER NOT NULL,
    input_tokens_used INTEGER NOT NULL,
    output_tokens_used INTEGER NOT NULL,
    total_tokens_used INTEGER NOT NULL,
    child_runs_used INTEGER NOT NULL,
    continuations_used INTEGER NOT NULL,
    active_runs INTEGER NOT NULL,
    stopped INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, root_run_id),
    FOREIGN KEY (namespace, root_run_id)
        REFERENCES runtime_runs (namespace, id) ON DELETE CASCADE
);
CREATE TABLE runtime_plugin_state (
    namespace TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    state_kind TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    payload BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (
        namespace, owner_id, scope, scope_id, state_kind, sequence
    )
)
"""
_RUN_COLUMNS = """
id, namespace, session_id, parent_run_id, root_run_id, relation, initiator,
input_digest, full_messages, active_messages, idempotency_key, prompt, status,
created_at, updated_at, owner_id, lease_expires_at, output, usage,
session_revision, error_code, error_detail
"""


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
            required_tables = {
                "runtime_metadata",
                "runtime_sessions",
                "runtime_runs",
                "runtime_run_events",
                "runtime_plugin_state",
                "runtime_root_budgets",
            }
            if not tables:
                connection.executescript(
                    f"""
                    BEGIN IMMEDIATE;
                    {_RUNTIME_METADATA_SCHEMA};
                    INSERT INTO runtime_metadata (singleton, schema_version)
                    VALUES (1, {_RUNTIME_SCHEMA_VERSION});
                    {_RUNTIME_SCHEMA};
                    COMMIT;
                    """
                )
            else:
                if not required_tables.issubset(tables):
                    raise RuntimeStoreSchemaError(
                        "Database contains an incomplete or unsupported runtime schema"
                    )
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
        return await _to_thread(self._create_session, namespace, pin)

    async def list_sessions(self, namespace: str) -> list[str]:
        return await _to_thread(self._list_runtime_sessions, namespace)

    async def load_session(
        self,
        namespace: str,
        session_id: str,
    ) -> RuntimeSessionSnapshot:
        return await _to_thread(self._load_runtime_session, namespace, session_id)

    async def reserve_run(
        self,
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
        return await _to_thread(
            self._reserve_run,
            namespace,
            session_id,
            idempotency_key,
            prompt,
            max_pending,
            parent_run_id,
            root_run_id,
            relation,
            root_budget,
            initiator,
            input_digest,
            full_messages,
            active_messages,
        )

    async def load_run(self, namespace: str, run_id: str) -> RuntimeRunSnapshot:
        return await _to_thread(self._load_run, namespace, run_id)

    async def load_root_budget(
        self,
        namespace: str,
        root_run_id: str,
    ) -> RootBudgetSnapshot | None:
        return await _to_thread(self._load_root_budget, namespace, root_run_id)

    async def list_runs(
        self,
        namespace: str,
        *,
        session_id: str | None = None,
    ) -> list[RuntimeRunSnapshot]:
        return await _to_thread(self._list_runs, namespace, session_id)

    async def cancel_run(
        self,
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
        return await _to_thread(
            self._cancel_run,
            namespace,
            run_id,
            error_code,
            error_detail,
            expected_session_revision,
            full_messages,
            active_messages,
            output,
            usage,
        )

    async def read_state(
        self,
        namespace: str,
        *,
        owner_id: str,
        scope: StateScope,
        scope_id: str,
        state_kind: str,
        schema_version: int | None = None,
    ) -> list[PluginStateEntry]:
        return await _to_thread(
            self._read_state,
            namespace,
            owner_id,
            scope,
            scope_id,
            state_kind,
            schema_version,
        )

    async def append_state(
        self,
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
        return await _to_thread(
            self._append_state,
            namespace,
            owner_id,
            scope,
            scope_id,
            state_kind,
            schema_version,
            expected_sequence,
            payload,
        )

    async def claim_run(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot:
        return await _to_thread(
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
        return await _to_thread(
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
        return await _to_thread(
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
        return await _to_thread(self._list_run_events, namespace, run_id, after)

    async def complete_run(
        self,
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
        return await _to_thread(
            self._complete_run,
            namespace,
            run_id,
            owner_id,
            expected_session_revision,
            messages,
            full_messages,
            active_messages,
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
        return await _to_thread(
            self._fail_run,
            namespace,
            run_id,
            owner_id,
            error_code,
            error_detail,
        )

    async def interrupt_owned_runs(self, owner_id: str) -> int:
        return await _to_thread(self._interrupt_owned_runs, owner_id)

    async def interrupt_expired_runs(self) -> int:
        return await _to_thread(self._interrupt_expired_runs)

    async def delete_session(
        self,
        namespace: str,
        session_id: str,
        *,
        expected_revision: str,
    ) -> None:
        await _to_thread(
            self._delete_runtime_session,
            namespace,
            session_id,
            expected_revision,
        )

    async def checkout_session(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        *,
        expected_revision: str,
    ) -> RuntimeSessionSnapshot:
        return await _to_thread(
            self._checkout_session,
            namespace,
            session_id,
            run_id,
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
        return await _to_thread(
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
                            namespace, id, agent_spec_id, model_id, revision, messages, head_run_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            validated_namespace,
                            session_id,
                            pin.agent_spec_id,
                            pin.model_id,
                            revision,
                            _EMPTY_MESSAGES,
                            None,
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
                head_run_id=None,
            )

    def _read_state(
        self,
        namespace: str,
        owner_id: str,
        scope: StateScope,
        scope_id: str,
        state_kind: str,
        schema_version: int | None,
    ) -> list[PluginStateEntry]:
        validated_namespace = validate_namespace(namespace)
        validated_owner = _validate_state_text(owner_id, "owner_id")
        validated_scope = _validate_state_scope(scope)
        validated_scope_id = validate_session_id(scope_id)
        validated_kind = _validate_state_text(state_kind, "state_kind")
        if schema_version is not None:
            _validate_schema_version(schema_version)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT owner_id, scope, scope_id, state_kind, schema_version,
                       sequence, payload, created_at
                FROM runtime_plugin_state
                WHERE namespace = ? AND owner_id = ? AND scope = ?
                  AND scope_id = ? AND state_kind = ?
                ORDER BY sequence
                """,
                (
                    validated_namespace,
                    validated_owner,
                    validated_scope.value,
                    validated_scope_id,
                    validated_kind,
                ),
            ).fetchall()
        entries = [
            _state_entry(
                namespace=validated_namespace,
                row=row,
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

    def _append_state(
        self,
        namespace: str,
        owner_id: str,
        scope: StateScope,
        scope_id: str,
        state_kind: str,
        schema_version: int,
        expected_sequence: int,
        payload: Any,
    ) -> PluginStateEntry:
        validated_namespace = validate_namespace(namespace)
        validated_owner = _validate_state_text(owner_id, "owner_id")
        validated_scope = _validate_state_scope(scope)
        validated_scope_id = validate_session_id(scope_id)
        validated_kind = _validate_state_text(state_kind, "state_kind")
        _validate_schema_version(schema_version)
        if expected_sequence < 0:
            raise ValueError("Expected state sequence cannot be negative")
        encoded = encode_state_payload(payload)
        stored_payload = decode_state_payload(encoded)
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0)
                FROM runtime_plugin_state
                WHERE namespace = ? AND owner_id = ? AND scope = ?
                  AND scope_id = ? AND state_kind = ?
                """,
                (
                    validated_namespace,
                    validated_owner,
                    validated_scope.value,
                    validated_scope_id,
                    validated_kind,
                ),
            ).fetchone()
            assert row is not None
            current_sequence = row[0]
            if current_sequence != expected_sequence:
                raise StateSequenceConflictError(
                    f"State stream {validated_kind!r} is at sequence {current_sequence}"
                )
            sequence = current_sequence + 1
            connection.execute(
                """
                INSERT INTO runtime_plugin_state (
                    namespace, owner_id, scope, scope_id, state_kind,
                    schema_version, sequence, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validated_namespace,
                    validated_owner,
                    validated_scope.value,
                    validated_scope_id,
                    validated_kind,
                    schema_version,
                    sequence,
                    encoded,
                    now.isoformat(),
                ),
            )
        return PluginStateEntry(
            namespace=validated_namespace,
            owner_id=validated_owner,
            scope=validated_scope,
            scope_id=validated_scope_id,
            state_kind=validated_kind,
            schema_version=schema_version,
            sequence=sequence,
            payload=stored_payload,
            created_at=now,
        )

    def _reserve_run(
        self,
        namespace: str,
        session_id: str,
        idempotency_key: str,
        prompt: str,
        max_pending: int | None,
        parent_run_id: str | None,
        root_run_id: str | None,
        relation: RunRelation,
        root_budget: RuntimeBudget | None,
        initiator: str,
        run_input_digest: str | None,
        full_messages: Sequence[ModelMessage],
        active_messages: Sequence[ModelMessage],
    ) -> RunReservation:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(session_id)
        if not idempotency_key or len(idempotency_key) > 255:
            raise ValueError("Idempotency key must contain between 1 and 255 characters")
        if not initiator.strip():
            raise ValueError("Run initiator must not be empty")
        if parent_run_id is not None:
            validate_session_id(parent_run_id)
        if root_run_id is not None:
            validate_session_id(root_run_id)
        try:
            relation = RunRelation(relation)
        except (TypeError, ValueError) as error:
            raise RunStateError("Run relation is unsupported") from error
        if relation is RunRelation.ROOT and (parent_run_id is not None or root_run_id is not None):
            raise RunStateError("A root Run cannot have an ancestor")
        if relation is not RunRelation.ROOT and parent_run_id is None:
            raise RunStateError("A related Run requires a parent Run")
        if relation is not RunRelation.ROOT and root_budget is not None:
            raise RunStateError("Only a root Run can define a root budget")
        run_input_digest = run_input_digest or input_digest(prompt)
        full_content = ModelMessagesTypeAdapter.dump_json(list(full_messages))
        active_content = ModelMessagesTypeAdapter.dump_json(list(active_messages))
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
                run = _runtime_run(row, connection=connection)
                if run.prompt != prompt or run.input_digest != run_input_digest:
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
            effective_root_run_id = run_id
            if relation is not RunRelation.ROOT:
                parent_row = connection.execute(
                    """
                    SELECT session_id, root_run_id FROM runtime_runs
                    WHERE namespace = ? AND id = ?
                    """,
                    (validated_namespace, parent_run_id),
                ).fetchone()
                if parent_row is None or parent_row[0] != validated_id:
                    raise RunNotFoundError(f"Parent Run {parent_run_id!r} does not exist")
                parent_root_run_id = parent_row[1] or parent_run_id
                assert parent_root_run_id is not None
                if root_run_id is not None and root_run_id != parent_root_run_id:
                    raise RunStateError("Related Run root does not match its parent")
                effective_root_run_id = parent_root_run_id
                root_budget_row = connection.execute(
                    """
                    SELECT request_limit, input_tokens_limit, output_tokens_limit,
                           total_tokens_limit, child_run_limit, continuation_limit,
                           wall_deadline, requests_used, input_tokens_used,
                           output_tokens_used, total_tokens_used, child_runs_used,
                           continuations_used, active_runs, stopped
                    FROM runtime_root_budgets
                    WHERE namespace = ? AND root_run_id = ?
                    """,
                    (validated_namespace, effective_root_run_id),
                ).fetchone()
                if root_budget_row is not None:
                    _admit_root_budget(
                        connection,
                        namespace=validated_namespace,
                        root_run_id=effective_root_run_id,
                        relation=relation,
                        row=root_budget_row,
                        now=now,
                    )
            connection.execute(
                """
                INSERT INTO runtime_runs (
                    namespace, id, session_id, parent_run_id, root_run_id, relation, initiator,
                    input_digest, full_messages, active_messages, idempotency_key, prompt, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validated_namespace,
                    run_id,
                    validated_id,
                    parent_run_id,
                    effective_root_run_id,
                    relation,
                    initiator,
                    run_input_digest,
                    full_content,
                    active_content,
                    idempotency_key,
                    prompt,
                    RunStatus.ACCEPTED,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            if root_budget is not None:
                _insert_root_budget(
                    connection,
                    namespace=validated_namespace,
                    root_run_id=run_id,
                    budget=root_budget,
                    now=now,
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
                parent_run_id=parent_run_id,
                root_run_id=effective_root_run_id,
                relation=relation,
                initiator=initiator,
                input_digest=run_input_digest,
                full_messages=list(full_messages),
                active_messages=list(active_messages),
            ),
            created=True,
        )

    def _load_run(self, namespace: str, run_id: str) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        with self._connect() as connection:
            return _select_runtime_run(connection, validated_namespace, validated_id)

    def _load_root_budget(
        self,
        namespace: str,
        root_run_id: str,
    ) -> RootBudgetSnapshot | None:
        validated_namespace = validate_namespace(namespace)
        validated_root_id = validate_session_id(root_run_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_limit, input_tokens_limit, output_tokens_limit,
                       total_tokens_limit, child_run_limit, continuation_limit,
                       wall_deadline, requests_used, input_tokens_used,
                       output_tokens_used, total_tokens_used, child_runs_used,
                       continuations_used, active_runs, stopped, updated_at
                FROM runtime_root_budgets
                WHERE namespace = ? AND root_run_id = ?
                """,
                (validated_namespace, validated_root_id),
            ).fetchone()
        if row is None:
            return None
        return _root_budget_snapshot(
            namespace=validated_namespace,
            root_run_id=validated_root_id,
            row=row,
        )

    def _list_runs(
        self,
        namespace: str,
        session_id: str | None,
    ) -> list[RuntimeRunSnapshot]:
        validated_namespace = validate_namespace(namespace)
        validated_session_id = validate_session_id(session_id) if session_id is not None else None
        query = f"SELECT {_RUN_COLUMNS} FROM runtime_runs WHERE namespace = ?"
        parameters: list[str] = [validated_namespace]
        if validated_session_id is not None:
            query += " AND session_id = ?"
            parameters.append(validated_session_id)
        query += " ORDER BY created_at, id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            return [_runtime_run(row, connection=connection) for row in rows]

    def _cancel_run(
        self,
        namespace: str,
        run_id: str,
        error_code: str,
        error_detail: str,
        expected_session_revision: str | None,
        full_messages: Sequence[ModelMessage] | None,
        active_messages: Sequence[ModelMessage] | None,
        output: Any | None,
        usage: RunUsage | None,
    ) -> RuntimeRunSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        if not error_code or not error_detail:
            raise ValueError("Run cancellation requires a stable code and safe detail")
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = _select_runtime_run(connection, validated_namespace, validated_id)
            if run.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.INTERRUPTED,
            }:
                return run
            session_row = connection.execute(
                """
                SELECT revision
                FROM runtime_sessions
                WHERE namespace = ? AND id = ?
                """,
                (validated_namespace, run.session_id),
            ).fetchone()
            if session_row is None:
                raise SessionNotFoundError(f"Session {run.session_id!r} does not exist")
            (current_revision,) = session_row
            if (
                expected_session_revision is not None
                and current_revision != expected_session_revision
            ):
                raise SessionConflictError(f"Session {run.session_id!r} has changed")
            full_messages = list(full_messages if full_messages is not None else run.full_messages)
            active_messages = list(
                active_messages if active_messages is not None else run.active_messages
            )
            full_content = ModelMessagesTypeAdapter.dump_json(full_messages)
            active_content = ModelMessagesTypeAdapter.dump_json(active_messages)
            session_revision = _revision(active_content)
            connection.execute(
                """
                UPDATE runtime_sessions SET revision = ?, messages = ?, head_run_id = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    session_revision,
                    active_content,
                    validated_id,
                    validated_namespace,
                    run.session_id,
                ),
            )
            connection.execute(
                """
                UPDATE runtime_runs
                SET status = ?, updated_at = ?, owner_id = NULL, lease_expires_at = NULL,
                    error_code = ?, error_detail = ?, session_revision = ?,
                    full_messages = ?, active_messages = ?, output = ?, usage = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    RunStatus.INTERRUPTED,
                    now.isoformat(),
                    error_code,
                    error_detail,
                    session_revision,
                    full_content,
                    active_content,
                    to_json(output) if output is not None else None,
                    _USAGE_ADAPTER.dump_json(usage) if usage is not None else None,
                    validated_namespace,
                    validated_id,
                ),
            )
            _account_root_budget(
                connection,
                namespace=validated_namespace,
                run=run,
                usage=usage,
                stopped=run.relation is RunRelation.ROOT,
                now=now,
            )
        return replace(
            run,
            status=RunStatus.INTERRUPTED,
            updated_at=now,
            owner_id=None,
            lease_expires_at=None,
            error_code=error_code,
            error_detail=error_detail,
            session_revision=session_revision,
            full_messages=full_messages,
            active_messages=active_messages,
            output=output,
            usage=usage,
        )

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
        messages: Sequence[ModelMessage] | None,
        full_messages: Sequence[ModelMessage] | None,
        active_messages: Sequence[ModelMessage] | None,
        output: Any,
        usage: RunUsage,
    ) -> RunCompletion:
        validated_namespace = validate_namespace(namespace)
        validated_id = validate_session_id(run_id)
        full_messages = list(full_messages if full_messages is not None else messages or ())
        active_messages = list(active_messages if active_messages is not None else messages or ())
        full_content = ModelMessagesTypeAdapter.dump_json(full_messages)
        active_content = ModelMessagesTypeAdapter.dump_json(active_messages)
        session_revision = _revision(active_content)
        output_content = to_json(output)
        usage_content = _USAGE_ADAPTER.dump_json(usage)
        now = datetime.now(UTC)
        run_events: list[StoredRunEvent] = []
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
                SELECT agent_spec_id, model_id, revision, head_run_id
                FROM runtime_sessions
                WHERE namespace = ? AND id = ?
                """,
                (validated_namespace, run.session_id),
            ).fetchone()
            if session_row is None:
                raise SessionNotFoundError(f"Session {run.session_id!r} does not exist")
            agent_spec_id, model_id, current_revision, _current_head_run_id = session_row
            if current_revision != expected_session_revision:
                raise SessionConflictError(f"Session {run.session_id!r} has changed")
            connection.execute(
                """
                UPDATE runtime_sessions SET revision = ?, messages = ?, head_run_id = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    session_revision,
                    active_content,
                    validated_id,
                    validated_namespace,
                    run.session_id,
                ),
            )
            connection.execute(
                """
                UPDATE runtime_runs
                SET status = ?, updated_at = ?, owner_id = NULL, lease_expires_at = NULL,
                    output = ?, usage = ?, session_revision = ?,
                    full_messages = ?, active_messages = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    RunStatus.COMPLETED,
                    now.isoformat(),
                    output_content,
                    usage_content,
                    session_revision,
                    full_content,
                    active_content,
                    validated_namespace,
                    validated_id,
                ),
            )
            _account_root_budget(
                connection,
                namespace=validated_namespace,
                run=run,
                usage=usage,
                stopped=False,
                now=now,
            )
            run_events = _run_events(connection, validated_namespace, validated_id)
        session = RuntimeSessionSnapshot(
            id=run.session_id,
            namespace=validated_namespace,
            pin=ExecutionPin(agent_spec_id=agent_spec_id, model_id=model_id),
            messages=list(active_messages),
            revision=session_revision,
            head_run_id=validated_id,
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
            full_messages=list(full_messages),
            active_messages=list(active_messages),
            events=run_events,
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
            _account_root_budget(
                connection,
                namespace=validated_namespace,
                run=run,
                usage=None,
                stopped=False,
                now=now,
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
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"SELECT {_RUN_COLUMNS} FROM runtime_runs WHERE status = ? AND owner_id = ?",
                (RunStatus.RUNNING, owner_id),
            ).fetchall()
            connection.execute(
                """
                UPDATE runtime_runs
                SET status = ?, updated_at = ?, owner_id = NULL, lease_expires_at = NULL,
                    error_code = ?, error_detail = ?
                WHERE status = ? AND owner_id = ?
                """,
                (
                    RunStatus.INTERRUPTED,
                    now.isoformat(),
                    "run_interrupted",
                    "Run worker stopped before completion",
                    RunStatus.RUNNING,
                    owner_id,
                ),
            )
            for row in rows:
                run = _runtime_run(row, connection=connection)
                _account_root_budget(
                    connection,
                    namespace=run.namespace,
                    run=run,
                    usage=None,
                    stopped=False,
                    now=now,
                )
            return len(rows)

    def _interrupt_expired_runs(self) -> int:
        now = datetime.now(UTC)
        now_text = now.isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"""
                SELECT {_RUN_COLUMNS} FROM runtime_runs
                WHERE status = ? AND lease_expires_at <= ?
                """,
                (RunStatus.RUNNING, now_text),
            ).fetchall()
            connection.execute(
                """
                UPDATE runtime_runs
                SET status = ?, updated_at = ?, owner_id = NULL, lease_expires_at = NULL,
                    error_code = ?, error_detail = ?
                WHERE status = ? AND lease_expires_at <= ?
                """,
                (
                    RunStatus.INTERRUPTED,
                    now_text,
                    "run_interrupted",
                    "Run worker stopped before completion",
                    RunStatus.RUNNING,
                    now_text,
                ),
            )
            for row in rows:
                run = _runtime_run(row, connection=connection)
                _account_root_budget(
                    connection,
                    namespace=run.namespace,
                    run=run,
                    usage=None,
                    stopped=False,
                    now=now,
                )
            return len(rows)

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
                SELECT agent_spec_id, model_id, revision, messages, head_run_id
                FROM runtime_sessions
                WHERE namespace = ? AND id = ?
                """,
                (validated_namespace, validated_id),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session {session_id!r} does not exist")
        agent_spec_id, model_id, revision, content, head_run_id = row
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
            head_run_id=head_run_id,
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
                """
                DELETE FROM runtime_plugin_state
                WHERE namespace = ? AND (
                    (scope = ? AND scope_id = ?)
                    OR (
                        scope = ? AND scope_id IN (
                            SELECT id FROM runtime_runs
                            WHERE namespace = ? AND session_id = ?
                        )
                    )
                )
                """,
                (
                    validated_namespace,
                    StateScope.SESSION,
                    validated_id,
                    StateScope.RUN,
                    validated_namespace,
                    validated_id,
                ),
            )
            connection.execute(
                "DELETE FROM runtime_sessions WHERE namespace = ? AND id = ?",
                (validated_namespace, validated_id),
            )

    def _checkout_session(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        expected_revision: str,
    ) -> RuntimeSessionSnapshot:
        validated_namespace = validate_namespace(namespace)
        validated_session_id = validate_session_id(session_id)
        validated_run_id = validate_session_id(run_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = _select_runtime_run(connection, validated_namespace, validated_run_id)
            if run.session_id != validated_session_id:
                raise RunStateError(f"Run {run_id!r} belongs to another session")
            if run.status not in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.INTERRUPTED,
            }:
                raise RunStateError(f"Run {run_id!r} is not terminal")
            row = connection.execute(
                """
                SELECT agent_spec_id, model_id, revision
                FROM runtime_sessions
                WHERE namespace = ? AND id = ?
                """,
                (validated_namespace, validated_session_id),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session {session_id!r} does not exist")
            agent_spec_id, model_id, current_revision = row
            if current_revision != expected_revision:
                raise SessionConflictError(f"Session {session_id!r} has changed")
            active_content = ModelMessagesTypeAdapter.dump_json(run.active_messages)
            revision = _revision(active_content)
            connection.execute(
                """
                UPDATE runtime_sessions SET revision = ?, messages = ?, head_run_id = ?
                WHERE namespace = ? AND id = ?
                """,
                (
                    revision,
                    active_content,
                    validated_run_id,
                    validated_namespace,
                    validated_session_id,
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
                SELECT agent_spec_id, model_id, revision, head_run_id
                FROM runtime_sessions
                WHERE namespace = ? AND id = ?
                """,
                (validated_namespace, validated_id),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session {session_id!r} does not exist")
            agent_spec_id, model_id, current_revision, head_run_id = row
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
            head_run_id=head_run_id,
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database, timeout=5)
        try:
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()


async def _to_thread[ResultT](
    function: Callable[..., ResultT],
    *args: Any,
) -> ResultT:
    """Run blocking store work to completion before propagating cancellation."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await asyncio.gather(worker, return_exceptions=True)
        raise


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
    return _runtime_run(row, connection=connection)


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


def _insert_root_budget(
    connection: sqlite3.Connection,
    *,
    namespace: str,
    root_run_id: str,
    budget: RuntimeBudget,
    now: datetime,
) -> None:
    if budget.wall_deadline is not None and budget.wall_deadline <= now:
        raise RunBudgetExceededError("Root wall deadline has passed")
    if budget.request_limit == 0:
        raise RunBudgetExceededError("Root request budget is exhausted")
    connection.execute(
        """
        INSERT INTO runtime_root_budgets (
            namespace, root_run_id, request_limit, input_tokens_limit,
            output_tokens_limit, total_tokens_limit, child_run_limit,
            continuation_limit, wall_deadline, requests_used,
            input_tokens_used, output_tokens_used, total_tokens_used,
            child_runs_used, continuations_used, active_runs, stopped, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0, 1, 0, ?)
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
            budget.wall_deadline.isoformat() if budget.wall_deadline is not None else None,
            now.isoformat(),
        ),
    )


def _admit_root_budget(
    connection: sqlite3.Connection,
    *,
    namespace: str,
    root_run_id: str,
    relation: RunRelation,
    row: sqlite3.Row | tuple[Any, ...],
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
        active_runs,
        stopped,
    ) = row
    del active_runs
    if stopped:
        raise RunBudgetExceededError("Root execution was cancelled")
    if wall_deadline is not None and datetime.fromisoformat(wall_deadline) <= now:
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
        counter = "child_runs_used"
    else:
        if continuation_limit is not None and continuations_used >= continuation_limit:
            raise RunBudgetExceededError("Root continuation budget is exhausted")
        counter = "continuations_used"
    connection.execute(
        f"""
        UPDATE runtime_root_budgets
        SET {counter} = {counter} + 1, active_runs = active_runs + 1, updated_at = ?
        WHERE namespace = ? AND root_run_id = ?
        """,
        (now.isoformat(), namespace, root_run_id),
    )


def _account_root_budget(
    connection: sqlite3.Connection,
    *,
    namespace: str,
    run: RuntimeRunSnapshot,
    usage: RunUsage | None,
    stopped: bool,
    now: datetime,
) -> None:
    if run.root_run_id is None:
        return
    usage = usage or RunUsage()
    connection.execute(
        """
        UPDATE runtime_root_budgets
        SET requests_used = requests_used + ?,
            input_tokens_used = input_tokens_used + ?,
            output_tokens_used = output_tokens_used + ?,
            total_tokens_used = total_tokens_used + ?,
            active_runs = CASE WHEN active_runs > 0 THEN active_runs - 1 ELSE 0 END,
            stopped = CASE WHEN ? THEN 1 ELSE stopped END,
            updated_at = ?
        WHERE namespace = ? AND root_run_id = ?
        """,
        (
            usage.requests,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            stopped,
            now.isoformat(),
            namespace,
            run.root_run_id,
        ),
    )


def _root_budget_snapshot(
    *,
    namespace: str,
    root_run_id: str,
    row: sqlite3.Row | tuple[Any, ...],
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
            wall_deadline=datetime.fromisoformat(wall_deadline)
            if wall_deadline is not None
            else None,
        ),
        requests_used=requests_used,
        input_tokens_used=input_tokens_used,
        output_tokens_used=output_tokens_used,
        total_tokens_used=total_tokens_used,
        child_runs_used=child_runs_used,
        continuations_used=continuations_used,
        active_runs=active_runs,
        stopped=bool(stopped),
        updated_at=datetime.fromisoformat(updated_at),
    )


def _validate_state_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"State {field_name} must not be empty")
    if len(value) > 128:
        raise ValueError(f"State {field_name} exceeds 128 characters")
    return value


def _validate_state_scope(scope: StateScope) -> StateScope:
    try:
        return StateScope(scope)
    except (TypeError, ValueError) as error:
        raise StateFormatError("State scope is unsupported") from error


def _validate_schema_version(schema_version: int) -> None:
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise StateSchemaError("State schema version must be an integer")
    if schema_version < 1:
        raise StateSchemaError("State schema version must be positive")


def _state_entry(
    *,
    namespace: str,
    row: sqlite3.Row | tuple[Any, ...],
) -> PluginStateEntry:
    owner_id, scope, scope_id, state_kind, schema_version, sequence, payload, created_at = row
    _validate_schema_version(schema_version)
    if not isinstance(sequence, int) or sequence < 1:
        raise StateFormatError("Stored state sequence is invalid")
    try:
        timestamp = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as error:
        raise StateFormatError("Stored state timestamp is invalid") from error
    return PluginStateEntry(
        namespace=namespace,
        owner_id=owner_id,
        scope=_validate_state_scope(scope),
        scope_id=scope_id,
        state_kind=state_kind,
        schema_version=schema_version,
        sequence=sequence,
        payload=decode_state_payload(payload),
        created_at=timestamp,
    )


def _runtime_run(
    row: sqlite3.Row | tuple[Any, ...],
    *,
    connection: sqlite3.Connection | None = None,
) -> RuntimeRunSnapshot:
    (
        run_id,
        namespace,
        session_id,
        parent_run_id,
        root_run_id,
        relation,
        initiator,
        input_digest,
        full_content,
        active_content,
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
    events = _run_events(connection, namespace, run_id) if connection is not None else []
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
        parent_run_id=parent_run_id,
        root_run_id=root_run_id,
        relation=RunRelation(relation),
        initiator=initiator,
        input_digest=input_digest,
        full_messages=ModelMessagesTypeAdapter.validate_json(full_content),
        active_messages=ModelMessagesTypeAdapter.validate_json(active_content),
        events=events,
    )


def _run_events(
    connection: sqlite3.Connection,
    namespace: str,
    run_id: str,
) -> list[StoredRunEvent]:
    rows = connection.execute(
        """
        SELECT sequence, event, data FROM runtime_run_events
        WHERE namespace = ? AND run_id = ?
        ORDER BY sequence
        """,
        (namespace, run_id),
    ).fetchall()
    return [StoredRunEvent(sequence=row[0], event=row[1], data=row[2]) for row in rows]


def _revision(content: bytes) -> str:
    return sha256(content).hexdigest()


__all__ = ["SQLiteRuntimeStore"]
