from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import Request
from httpx import ASGITransport, AsyncClient
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_webserver import create_app

SPEC_PATH = Path(__file__).parents[1] / "examples" / "agents" / "chat.yaml"


def local_namespace(_request: Request) -> str:
    return "local"


def no_deps(_namespace: str, _session_id: str) -> None:
    return None


async def test_chat_agentspec_runs_through_the_streaming_web_host(tmp_path: Path) -> None:
    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "Web chat works"

    spec = AgentSpec.from_file(SPEC_PATH)
    app = create_app(
        spec,
        SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:function",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session_id = (await client.post("/v1/sessions")).json()["id"]
        response = await client.post(
            f"/v1/sessions/{session_id}/runs/stream",
            headers={"Idempotency-Key": "request-1"},
            json={"prompt": "Hello"},
        )

    assert spec.model == "openai:gpt-5.6-luna"
    assert response.status_code == 200
    assert "Web chat works" in response.text
    assert "event: run_completed" in response.text
