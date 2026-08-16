import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn
from fastapi import Request
from pydantic_ai import AgentSpec
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_workspace_session import WorkspaceRuntimeStore

from . import create_app


def main(argv: Sequence[str] | None = None) -> None:
    """Run one AgentSpec through the YoloP FastAPI host."""
    args = _parser().parse_args(argv)
    spec = AgentSpec.from_file(args.agent_spec)
    session_path = args.session_path
    if args.session_backend == "sqlite":
        store = SQLiteRuntimeStore(session_path or Path(".yolop/runtime.db"))
    else:
        store = WorkspaceRuntimeStore(session_path or Path.cwd())
    app = create_app(
        spec,
        store,
        namespace_resolver=_local_namespace,
        deps_resolver=_no_deps,
        deps_type=type(None),
    )
    uvicorn.run(app, host=args.host, port=args.port)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-spec", required=True, type=Path)
    parser.add_argument(
        "--session-backend",
        choices=("sqlite", "workspace"),
        default="sqlite",
    )
    parser.add_argument("--session-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def _local_namespace(_request: Request) -> str:
    return "local"


def _no_deps(_namespace: str, _session_id: str) -> None:
    return None


if __name__ == "__main__":
    main()
