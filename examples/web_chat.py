import argparse
from pathlib import Path

import uvicorn
from fastapi import Request
from fastapi.responses import FileResponse
from pydantic_ai import AgentSpec
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_webserver import create_app

EXAMPLES = Path(__file__).parent
DEFAULT_SPEC = EXAMPLES / "agents" / "chat.yaml"
CHAT_PAGE = EXAMPLES / "web_chat.html"


def local_namespace(_request: Request) -> str:
    return "local"


def no_deps(_namespace: str, _session_id: str) -> None:
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--database", type=Path, default=Path(".yolop/web-chat.db"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        AgentSpec.from_file(args.agent_spec),
        SQLiteRuntimeStore(args.database),
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
    )

    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(CHAT_PAGE)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
