import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from yolop_session import ExecutionPin
from yolop_workspace_session import WorkspaceRuntimeStore

from yolop import Yolop

SPEC_PATH = Path(__file__).parents[1] / "examples" / "agents" / "duckdb.yaml"


@dataclass(frozen=True)
class HostDeps:
    duckdb_connection: duckdb.DuckDBPyConnection


async def test_duckdb_agentspec_runs_with_a_persistent_session(tmp_path: Path) -> None:
    spec = AgentSpec.from_file(SPEC_PATH)
    assert spec.model == "openai:gpt-5.6-luna"

    connection = duckdb.connect(":memory:", config={"enable_external_access": "false"})
    connection.execute("create table sales(amount integer)")
    connection.execute("insert into sales values (10), (20)")
    store = WorkspaceRuntimeStore(tmp_path)
    assert isinstance(spec.model, str)
    session = await store.create_session(
        "local",
        pin=ExecutionPin.from_spec(spec, model_id=spec.model),
    )

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
                    json_args=json.dumps({"sql": "select sum(amount) as total from sales"}),
                    tool_call_id="total-sales",
                )
            }
        else:
            assert tool_returns[-1].content == {
                "columns": ["total"],
                "rows": [[30]],
                "truncated": False,
            }
            yield "Total sales are 30"

    try:
        async with Yolop().run(
            spec,
            "Calculate total sales.",
            model=FunctionModel(stream_function=respond),
            deps=HostDeps(duckdb_connection=connection),
            deps_type=HostDeps,
            message_history=session.messages,
        ) as run:
            _ = [event async for event in run]
    finally:
        connection.close()

    saved = await store.replace_session(
        "local",
        session.id,
        expected_revision=session.revision,
        messages=run.all_messages(),
    )
    assert await store.load_session("local", session.id) == saved
