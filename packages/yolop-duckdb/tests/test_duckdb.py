import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
from pydantic_ai import AgentRunResultEvent, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pytest import mark, raises

from yolop import Yolop


@dataclass(frozen=True)
class HostDeps:
    duckdb_connection: duckdb.DuckDBPyConnection


def read_only_connection(
    tmp_path: Path,
    setup_sql: str = "",
    *,
    external_access: bool = False,
) -> duckdb.DuckDBPyConnection:
    database = tmp_path / "analytics.duckdb"
    connection = duckdb.connect(str(database))
    try:
        if setup_sql:
            connection.execute(setup_sql)
    finally:
        connection.close()
    if external_access:
        return duckdb.connect(str(database), read_only=True)
    return duckdb.connect(
        str(database),
        read_only=True,
        config={"enable_external_access": "false"},
    )


def test_max_rows_must_be_positive() -> None:
    with raises(ValueError, match="max_rows must be positive"):
        Yolop().run(
            {"capabilities": [{"DuckDB": {"max_rows": 0}}]},
            "Query data.",
            model=FunctionModel(stream_function=_unused_stream),
            deps=None,
            deps_type=type(None),
        )


async def _unused_stream(
    _messages: list[ModelMessage], _info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    yield "unused"


async def test_agent_can_query_a_host_provided_connection_with_a_row_limit(
    tmp_path: Path,
) -> None:
    connection = read_only_connection(
        tmp_path,
        "create table numbers(value integer); insert into numbers values (1), (2), (3)",
    )

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            assert {tool.name for tool in info.function_tools} == {"query_duckdb"}
            yield {
                0: DeltaToolCall(
                    name="query_duckdb",
                    json_args=json.dumps({"sql": "select value from numbers order by value"}),
                    tool_call_id="query",
                )
            }
        else:
            assert tool_returns[-1].content == {
                "columns": ["value"],
                "rows": [[1], [2]],
                "truncated": True,
            }
            yield "Query complete"

    try:
        async with Yolop().run(
            {"capabilities": [{"DuckDB": {"max_rows": 2}}]},
            "Query the numbers.",
            model=FunctionModel(stream_function=respond),
            deps=HostDeps(duckdb_connection=connection),
            deps_type=HostDeps,
        ) as run:
            events = [event async for event in run]
    finally:
        connection.close()

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Query complete"


@mark.parametrize(
    ("sql", "expected_error"),
    [
        ("insert into numbers values (1)", "Only read-only DuckDB queries are allowed"),
        ("select 1; select 2", "Provide exactly one DuckDB SQL statement"),
    ],
)
async def test_agent_cannot_run_unsafe_sql(
    tmp_path: Path,
    sql: str,
    expected_error: str,
) -> None:
    connection = read_only_connection(tmp_path, "create table numbers(value integer)")

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        retries = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        ]
        if not retries:
            yield {
                0: DeltaToolCall(
                    name="query_duckdb",
                    json_args=json.dumps({"sql": sql}),
                    tool_call_id="unsafe-query",
                )
            }
        else:
            assert expected_error in str(retries[-1].content)
            yield "Query rejected"

    try:
        async with Yolop().run(
            {"capabilities": ["DuckDB"]},
            "Run unsafe SQL.",
            model=FunctionModel(stream_function=respond),
            deps=HostDeps(duckdb_connection=connection),
            deps_type=HostDeps,
        ) as run:
            events = [event async for event in run]
        assert connection.execute("select count(*) from numbers").fetchone() == (0,)
    finally:
        connection.close()

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Query rejected"


async def test_connection_must_be_opened_read_only() -> None:
    connection = duckdb.connect(":memory:", config={"enable_external_access": "false"})
    connection.execute("create sequence counter")
    try:
        with raises(UserError, match="access_mode=read_only"):
            async with Yolop().run(
                {"capabilities": ["DuckDB"]},
                "Query data.",
                model=FunctionModel(stream_function=_unused_stream),
                deps=HostDeps(duckdb_connection=connection),
                deps_type=HostDeps,
            ) as run:
                _ = [event async for event in run]
        assert connection.execute("select nextval('counter')").fetchone() == (1,)
    finally:
        connection.close()


async def test_connection_must_disable_external_access(tmp_path: Path) -> None:
    connection = read_only_connection(tmp_path, external_access=True)
    try:
        with raises(UserError, match="enable_external_access=false"):
            async with Yolop().run(
                {"capabilities": ["DuckDB"]},
                "Query data.",
                model=FunctionModel(stream_function=_unused_stream),
                deps=HostDeps(duckdb_connection=connection),
                deps_type=HostDeps,
            ) as run:
                _ = [event async for event in run]
    finally:
        connection.close()


async def test_parallel_tool_calls_share_one_host_connection_safely(tmp_path: Path) -> None:
    connection = read_only_connection(tmp_path)

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            yield {
                0: DeltaToolCall(
                    name="query_duckdb",
                    json_args=json.dumps({"sql": "select 1 as value"}),
                    tool_call_id="first-query",
                ),
                1: DeltaToolCall(
                    name="query_duckdb",
                    json_args=json.dumps({"sql": "select 2 as value"}),
                    tool_call_id="second-query",
                ),
            }
        else:
            values: set[int] = set()
            for result in tool_returns:
                assert isinstance(result.content, dict)
                values.add(result.content["rows"][0][0])
            assert values == {1, 2}
            yield "Both queries complete"

    try:
        async with Yolop().run(
            {"capabilities": ["DuckDB"]},
            "Run two queries.",
            model=FunctionModel(stream_function=respond),
            deps=HostDeps(duckdb_connection=connection),
            deps_type=HostDeps,
        ) as run:
            events = [event async for event in run]
    finally:
        connection.close()

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Both queries complete"
