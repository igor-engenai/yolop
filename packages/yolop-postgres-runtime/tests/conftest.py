from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from yolop_postgres_runtime import migrate


def _database_conninfo(dsn: str, database: str) -> str:
    parameters = {
        key: str(value) for key, value in conninfo_to_dict(dsn).items() if value is not None
    }
    parameters["dbname"] = database
    return make_conninfo(**parameters)


@pytest.fixture
async def postgres_dsn() -> AsyncIterator[str]:
    admin_dsn = os.getenv("YOLOP_POSTGRES_TEST_DSN")
    if not admin_dsn:
        pytest.skip("YOLOP_POSTGRES_TEST_DSN is required for PostgreSQL integration tests")

    database = f"yolop_test_{uuid4().hex}"
    async with await psycopg.AsyncConnection.connect(admin_dsn, autocommit=True) as connection:
        await connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    dsn = _database_conninfo(admin_dsn, database)
    try:
        assert await migrate(dsn) == 1
        yield dsn
    finally:
        cleanup_dsn = _database_conninfo(admin_dsn, "postgres")
        async with await psycopg.AsyncConnection.connect(
            cleanup_dsn,
            autocommit=True,
        ) as connection:
            await connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )
