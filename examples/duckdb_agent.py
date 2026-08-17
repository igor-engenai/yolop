import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
from pydantic_ai import AgentSpec
from yolop_runtime import ExecutionPin, ensure_session_pin
from yolop_workspace_session import WorkspaceRuntimeStore

from yolop import Yolop

SPEC_PATH = Path(__file__).with_name("agents") / "duckdb.yaml"


@dataclass(frozen=True)
class HostDeps:
    duckdb_connection: duckdb.DuckDBPyConnection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--session", help="Resume a generated session ID")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    database = args.database.expanduser().resolve()
    if not database.is_file():
        raise SystemExit(f"DuckDB database does not exist: {database}")

    store = WorkspaceRuntimeStore(Path.cwd())
    spec = AgentSpec.from_file(SPEC_PATH)
    assert isinstance(spec.model, str)
    pin = ExecutionPin.from_spec(spec, model_id=spec.model)
    session = (
        await store.load_session("local", args.session)
        if args.session
        else await store.create_session("local", pin=pin)
    )
    ensure_session_pin(session, pin)
    print(f"session: {session.id}")

    connection = duckdb.connect(
        str(database),
        read_only=True,
        config={"enable_external_access": "false"},
    )
    try:
        async with Yolop().run(
            spec,
            args.prompt,
            deps=HostDeps(duckdb_connection=connection),
            deps_type=HostDeps,
            message_history=session.messages,
        ) as run:
            async for _ in run:
                pass
    finally:
        connection.close()

    assert run.result is not None
    await store.replace_session(
        "local",
        session.id,
        expected_revision=session.revision,
        messages=run.all_messages(),
    )
    print(run.result.output)


if __name__ == "__main__":
    asyncio.run(main())
