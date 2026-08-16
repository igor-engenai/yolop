import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn
from pydantic_ai import AgentSpec
from yolop_sqlite_session import SQLiteSessionStore
from yolop_workspace_session import WorkspaceSessionStore

from . import create_app


def main(argv: Sequence[str] | None = None) -> None:
    """Run one AgentSpec through the YoloP FastAPI host."""
    args = _parser().parse_args(argv)
    spec = AgentSpec.from_file(args.agent_spec)
    session_path = args.session_path
    if args.session_backend == "sqlite":
        store = SQLiteSessionStore(session_path or Path(".yolop/sessions.db"))
    else:
        store = WorkspaceSessionStore(session_path or Path.cwd())
    app = create_app(spec, store, deps=None, deps_type=type(None))
    uvicorn.run(app, host=args.host, port=args.port)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-spec", required=True, type=Path)
    parser.add_argument(
        "--session-backend",
        choices=("sqlite", "jsonl"),
        default="sqlite",
    )
    parser.add_argument("--session-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


if __name__ == "__main__":
    main()
