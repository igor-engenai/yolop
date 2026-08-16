from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

import duckdb
from pydantic_ai import ModelRetry, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset


class DuckDBDeps(Protocol):
    """Host dependencies required by the DuckDB capability."""

    @property
    def duckdb_connection(self) -> duckdb.DuckDBPyConnection: ...


@dataclass
class DuckDB(AbstractCapability[Any]):
    """Provide read-only SQL access to a host-opened DuckDB connection."""

    max_rows: int = 200

    def __post_init__(self) -> None:
        if self.max_rows < 1:
            raise ValueError("max_rows must be positive")

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        connection = _connection(ctx.deps)
        external_access = await asyncio.to_thread(_external_access_enabled, connection)
        if external_access:
            raise UserError(
                "DuckDB capability requires a connection with enable_external_access=false"
            )
        access_mode = await asyncio.to_thread(_access_mode, connection)
        if access_mode != "read_only":
            raise UserError("DuckDB capability requires a connection with access_mode=read_only")
        return _DuckDBRun(connection=connection, max_rows=self.max_rows)


@dataclass
class _DuckDBRun(AbstractCapability[Any]):
    connection: duckdb.DuckDBPyConnection
    max_rows: int
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _toolset: FunctionToolset[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._toolset = FunctionToolset(
            [Tool(self.query_duckdb, takes_ctx=False)],
            id="duckdb",
        )

    def get_instructions(self) -> str:
        return (
            "Use query_duckdb for one read-only DuckDB SQL query. "
            "Use information_schema when you need to inspect tables or columns."
        )

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    async def query_duckdb(self, sql: str) -> dict[str, Any]:
        """Run one read-only SQL query against the host-provided DuckDB database."""
        async with self._lock:
            return await asyncio.to_thread(_query, self.connection, sql, self.max_rows)


def _connection(deps: Any) -> duckdb.DuckDBPyConnection:
    value = getattr(deps, "duckdb_connection", None)
    if not isinstance(value, duckdb.DuckDBPyConnection):
        raise UserError(
            "DuckDB capability requires deps.duckdb_connection to be a DuckDB connection"
        )
    return value


def _external_access_enabled(connection: duckdb.DuckDBPyConnection) -> bool:
    cursor = connection.cursor()
    try:
        row = cursor.execute("select current_setting('enable_external_access')").fetchone()
        return bool(row and row[0])
    finally:
        cursor.close()


def _access_mode(connection: duckdb.DuckDBPyConnection) -> str | None:
    cursor = connection.cursor()
    try:
        row = cursor.execute("select current_setting('access_mode')").fetchone()
        return str(row[0]) if row else None
    finally:
        cursor.close()


def _query(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    max_rows: int,
) -> dict[str, Any]:
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as error:
        raise ModelRetry(f"Invalid DuckDB SQL: {error}") from error
    if len(statements) != 1:
        raise ModelRetry("Provide exactly one DuckDB SQL statement")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise ModelRetry("Only read-only DuckDB queries are allowed")

    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchmany(max_rows + 1)
    except duckdb.Error as error:
        raise ModelRetry(f"DuckDB query failed: {error}") from error
    finally:
        cursor.close()

    return {
        "columns": columns,
        "rows": [list(row) for row in rows[:max_rows]],
        "truncated": len(rows) > max_rows,
    }


__all__ = ["DuckDB", "DuckDBDeps"]
