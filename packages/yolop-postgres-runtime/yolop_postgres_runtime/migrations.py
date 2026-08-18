from __future__ import annotations

from dataclasses import dataclass
from typing import Final, LiteralString, cast

import psycopg
from psycopg import sql

CURRENT_SCHEMA_VERSION: Final = 1
_MIGRATION_LOCK_KEY = "yolop-postgres-runtime:migrations"


@dataclass(frozen=True)
class Migration:
    """One ordered PostgreSQL schema migration."""

    version: int
    name: str
    sql: str


_INITIAL_SCHEMA = """
CREATE TABLE yolop_runtime_sessions (
    namespace TEXT NOT NULL,
    id UUID NOT NULL,
    agent_spec_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    head_run_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (namespace, id)
);

CREATE TABLE yolop_runtime_session_messages (
    namespace TEXT NOT NULL,
    session_id UUID NOT NULL,
    sequence BIGINT NOT NULL CHECK (sequence > 0),
    message JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (namespace, session_id, sequence),
    FOREIGN KEY (namespace, session_id)
        REFERENCES yolop_runtime_sessions (namespace, id)
        ON DELETE CASCADE
);

CREATE TABLE yolop_runtime_session_contexts (
    namespace TEXT NOT NULL,
    session_id UUID NOT NULL,
    revision TEXT NOT NULL,
    messages JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (namespace, session_id, revision),
    FOREIGN KEY (namespace, session_id)
        REFERENCES yolop_runtime_sessions (namespace, id)
        ON DELETE CASCADE
);

CREATE TABLE yolop_runtime_runs (
    namespace TEXT NOT NULL,
    id UUID NOT NULL,
    session_id UUID NOT NULL,
    parent_run_id UUID,
    root_run_id UUID NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN ('root', 'continuation', 'child')),
    initiator TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    full_message_start BIGINT,
    full_message_end BIGINT,
    active_message_start BIGINT,
    active_message_end BIGINT,
    idempotency_key TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('accepted', 'running', 'completed', 'failed', 'interrupted')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    owner_id TEXT,
    lease_expires_at TIMESTAMPTZ,
    output JSONB,
    usage JSONB,
    session_revision TEXT,
    error_code TEXT,
    error_detail TEXT,
    PRIMARY KEY (namespace, id),
    UNIQUE (namespace, session_id, idempotency_key),
    FOREIGN KEY (namespace, session_id)
        REFERENCES yolop_runtime_sessions (namespace, id)
        ON DELETE CASCADE,
    FOREIGN KEY (namespace, parent_run_id)
        REFERENCES yolop_runtime_runs (namespace, id)
        ON DELETE RESTRICT,
    FOREIGN KEY (namespace, root_run_id)
        REFERENCES yolop_runtime_runs (namespace, id)
        ON DELETE RESTRICT,
    CHECK (
        (relation = 'root' AND parent_run_id IS NULL AND root_run_id = id)
        OR (relation <> 'root' AND parent_run_id IS NOT NULL)
    ),
    CHECK (
        (full_message_start IS NULL AND full_message_end IS NULL)
        OR (full_message_start IS NOT NULL AND full_message_end IS NOT NULL
            AND full_message_start <= full_message_end)
    ),
    CHECK (
        (active_message_start IS NULL AND active_message_end IS NULL)
        OR (active_message_start IS NOT NULL AND active_message_end IS NOT NULL
            AND active_message_start <= active_message_end)
    )
);

ALTER TABLE yolop_runtime_sessions
    ADD CONSTRAINT runtime_sessions_head_run_fk
    FOREIGN KEY (namespace, head_run_id)
    REFERENCES yolop_runtime_runs (namespace, id)
    ON DELETE SET NULL;

CREATE TABLE yolop_runtime_run_events (
    namespace TEXT NOT NULL,
    run_id UUID NOT NULL,
    sequence BIGINT NOT NULL CHECK (sequence > 0),
    event TEXT NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (namespace, run_id, sequence),
    FOREIGN KEY (namespace, run_id)
        REFERENCES yolop_runtime_runs (namespace, id)
        ON DELETE CASCADE
);

CREATE TABLE yolop_runtime_plugin_state (
    namespace TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('session', 'run')),
    scope_id UUID NOT NULL,
    state_kind TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    sequence BIGINT NOT NULL CHECK (sequence > 0),
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (
        namespace, owner_id, scope, scope_id, state_kind, sequence
    )
);

CREATE TABLE yolop_runtime_root_budgets (
    namespace TEXT NOT NULL,
    root_run_id UUID NOT NULL,
    request_limit INTEGER CHECK (request_limit IS NULL OR request_limit >= 0),
    input_tokens_limit INTEGER
        CHECK (input_tokens_limit IS NULL OR input_tokens_limit >= 0),
    output_tokens_limit INTEGER
        CHECK (output_tokens_limit IS NULL OR output_tokens_limit >= 0),
    total_tokens_limit INTEGER
        CHECK (total_tokens_limit IS NULL OR total_tokens_limit >= 0),
    child_run_limit INTEGER CHECK (child_run_limit IS NULL OR child_run_limit >= 0),
    continuation_limit INTEGER
        CHECK (continuation_limit IS NULL OR continuation_limit >= 0),
    wall_deadline TIMESTAMPTZ,
    requests_used INTEGER NOT NULL DEFAULT 0 CHECK (requests_used >= 0),
    input_tokens_used INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens_used >= 0),
    output_tokens_used INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens_used >= 0),
    total_tokens_used INTEGER NOT NULL DEFAULT 0 CHECK (total_tokens_used >= 0),
    child_runs_used INTEGER NOT NULL DEFAULT 0 CHECK (child_runs_used >= 0),
    continuations_used INTEGER NOT NULL DEFAULT 0 CHECK (continuations_used >= 0),
    active_runs INTEGER NOT NULL DEFAULT 0 CHECK (active_runs >= 0),
    stopped BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (namespace, root_run_id),
    FOREIGN KEY (namespace, root_run_id)
        REFERENCES yolop_runtime_runs (namespace, id)
        ON DELETE CASCADE
);

CREATE INDEX runtime_sessions_namespace_updated_idx
    ON yolop_runtime_sessions (namespace, updated_at, id);
CREATE INDEX runtime_runs_eligible_idx
    ON yolop_runtime_runs (namespace, status, created_at, id);
CREATE INDEX runtime_runs_session_status_idx
    ON yolop_runtime_runs (namespace, session_id, status, created_at, id);
CREATE INDEX runtime_runs_lease_idx
    ON yolop_runtime_runs (status, lease_expires_at)
    WHERE status = 'running';
CREATE INDEX runtime_run_events_cursor_idx
    ON yolop_runtime_run_events (namespace, run_id, sequence);
CREATE INDEX runtime_plugin_state_read_idx
    ON yolop_runtime_plugin_state (
        namespace, owner_id, scope, scope_id, state_kind, sequence
    );
CREATE INDEX runtime_root_budgets_active_idx
    ON yolop_runtime_root_budgets (namespace, active_runs, updated_at);
"""

MIGRATIONS: Final[tuple[Migration, ...]] = (
    Migration(version=1, name="initial_runtime_schema", sql=_INITIAL_SCHEMA),
)


async def migrate(dsn: str, *, target_version: int | None = None) -> int:
    """Apply explicit PostgreSQL migrations and return the installed version."""
    if not dsn.strip():
        raise ValueError("PostgreSQL DSN must not be empty")
    target = CURRENT_SCHEMA_VERSION if target_version is None else target_version
    if target < 0 or target > CURRENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported target schema version: {target}")

    async with await psycopg.AsyncConnection.connect(dsn) as connection:
        async with connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (_MIGRATION_LOCK_KEY,),
            )
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS yolop_runtime_schema_migrations (
                    version INTEGER PRIMARY KEY CHECK (version > 0),
                    name TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor = await connection.execute(
                "SELECT version FROM yolop_runtime_schema_migrations ORDER BY version DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            current = 0 if row is None else row[0]
            if current > target:
                raise ValueError(f"Database schema version {current} is newer than target {target}")

            for migration in MIGRATIONS:
                if migration.version <= current or migration.version > target:
                    continue
                await connection.execute(sql.SQL(cast(LiteralString, migration.sql)))
                await connection.execute(
                    """
                    INSERT INTO yolop_runtime_schema_migrations (version, name)
                    VALUES (%s, %s)
                    """,
                    (migration.version, migration.name),
                )
                current = migration.version
    return current


__all__ = ["CURRENT_SCHEMA_VERSION", "MIGRATIONS", "Migration", "migrate"]
