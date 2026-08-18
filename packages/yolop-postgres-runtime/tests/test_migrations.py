from __future__ import annotations

import os
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from yolop_postgres_runtime import CURRENT_SCHEMA_VERSION, migrate


def _database_conninfo(dsn: str, database: str) -> str:
    parameters = {
        key: str(value) for key, value in conninfo_to_dict(dsn).items() if value is not None
    }
    parameters["dbname"] = database
    return make_conninfo(**parameters)


async def _temporary_database(admin_dsn: str) -> tuple[str, str]:
    database = f"yolop_test_{uuid4().hex}"
    async with await psycopg.AsyncConnection.connect(admin_dsn, autocommit=True) as connection:
        await connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    return _database_conninfo(admin_dsn, database), database


@pytest.mark.asyncio
async def test_empty_postgresql_database_upgrades_idempotently() -> None:
    admin_dsn = os.getenv("YOLOP_POSTGRES_TEST_DSN")
    if not admin_dsn:
        pytest.skip("YOLOP_POSTGRES_TEST_DSN is required for PostgreSQL integration tests")

    dsn, database = await _temporary_database(admin_dsn)
    try:
        assert await migrate(dsn) == CURRENT_SCHEMA_VERSION
        assert await migrate(dsn) == CURRENT_SCHEMA_VERSION

        async with await psycopg.AsyncConnection.connect(dsn) as connection:
            cursor = await connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name LIKE 'yolop_runtime_%'
                ORDER BY table_name
                """
            )
            tables = {row[0] for row in await cursor.fetchall()}

        assert {
            "yolop_runtime_schema_migrations",
            "yolop_runtime_sessions",
            "yolop_runtime_session_messages",
            "yolop_runtime_session_contexts",
            "yolop_runtime_runs",
            "yolop_runtime_run_events",
            "yolop_runtime_plugin_state",
            "yolop_runtime_root_budgets",
        } <= tables
    finally:
        cleanup_dsn = _database_conninfo(admin_dsn, "postgres")
        async with await psycopg.AsyncConnection.connect(
            cleanup_dsn,
            autocommit=True,
        ) as connection:
            await connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )
