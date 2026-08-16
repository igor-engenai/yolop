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
from pydantic_core import to_json


class DuckDBDeps(Protocol):
    """Host dependencies required by the DuckDB capability."""

    @property
    def duckdb_connection(self) -> duckdb.DuckDBPyConnection: ...


@dataclass
class DuckDB(AbstractCapability[Any]):
    """Provide read-only SQL access to a host-opened DuckDB connection."""

    max_rows: int = 200
    max_result_bytes: int = 1_048_576
    timeout_seconds: float = 30

    def __post_init__(self) -> None:
        if self.max_rows < 1:
            raise ValueError("max_rows must be positive")
        if self.max_result_bytes < 1:
            raise ValueError("max_result_bytes must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

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
        return _DuckDBRun(
            connection=connection,
            max_rows=self.max_rows,
            max_result_bytes=self.max_result_bytes,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass
class _DuckDBRun(AbstractCapability[Any]):
    connection: duckdb.DuckDBPyConnection
    max_rows: int
    max_result_bytes: int
    timeout_seconds: float
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
            worker = asyncio.create_task(
                asyncio.to_thread(
                    _query,
                    self.connection,
                    sql,
                    self.max_rows,
                    self.max_result_bytes,
                )
            )
            try:
                return await asyncio.wait_for(
                    asyncio.shield(worker),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                self.connection.interrupt()
                await asyncio.gather(worker, return_exceptions=True)
                raise ModelRetry(
                    f"DuckDB query timed out after {self.timeout_seconds:g} seconds"
                ) from None
            except asyncio.CancelledError:
                self.connection.interrupt()
                await asyncio.gather(worker, return_exceptions=True)
                raise


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
    max_result_bytes: int,
) -> dict[str, Any]:
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as error:
        raise ModelRetry(f"Invalid DuckDB SQL: {error}") from error
    if len(statements) != 1:
        raise ModelRetry("Provide exactly one DuckDB SQL statement")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise ModelRetry("Only read-only DuckDB queries are allowed")

    try:
        connection.execute(sql)
        columns = [description[0] for description in connection.description]
        result_bytes = len(to_json({"columns": columns, "rows": [], "truncated": False}))
        if result_bytes > max_result_bytes:
            raise ModelRetry(f"DuckDB result exceeded {max_result_bytes} bytes; request less data")
        rows: list[list[Any]] = []
        truncated = False
        for index in range(max_rows + 1):
            raw_row = connection.fetchone()
            if raw_row is None:
                break
            if index == max_rows:
                truncated = True
                break
            row = list(raw_row)
            result_bytes += len(to_json(row)) + (1 if rows else 0)
            if result_bytes > max_result_bytes:
                raise ModelRetry(
                    f"DuckDB result exceeded {max_result_bytes} bytes; request less data"
                )
            rows.append(row)
    except duckdb.Error as error:
        raise ModelRetry(f"DuckDB query failed: {error}") from error

    return {
        "columns": columns,
        "rows": rows,
        "truncated": truncated,
    }


__all__ = ["DuckDB", "DuckDBDeps"]
