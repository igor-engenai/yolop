from pathlib import Path
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from yolop_webserver.cli import main


async def test_cli_loads_agentspec_and_uses_loopback_sqlite_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec_path = tmp_path / "agent.yaml"
    spec_path.write_text("name: web-agent\nmodel: openai:test\ninstructions: Be concise.\n")
    started: dict[str, Any] = {}

    def start(app: FastAPI, *, host: str, port: int) -> None:
        started.update(app=app, host=host, port=port)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("uvicorn.run", start)

    main(["--agent-spec", str(spec_path)])

    assert started["host"] == "127.0.0.1"
    assert started["port"] == 8000
    async with AsyncClient(
        transport=ASGITransport(app=started["app"]),
        base_url="http://test",
    ) as client:
        response = await client.post("/v1/sessions")
    assert response.status_code == 201
    assert (tmp_path / ".yolop" / "runtime.db").is_file()


async def test_cli_can_select_workspace_runtime_store(tmp_path: Path, monkeypatch) -> None:
    spec_path = tmp_path / "agent.yaml"
    spec_path.write_text("name: web-agent\nmodel: openai:test\n")
    workspace = tmp_path / "workspace"
    started: dict[str, Any] = {}

    def start(app: FastAPI, *, host: str, port: int) -> None:
        started.update(app=app, host=host, port=port)

    monkeypatch.setattr("uvicorn.run", start)
    main(
        [
            "--agent-spec",
            str(spec_path),
            "--session-backend",
            "workspace",
            "--session-path",
            str(workspace),
            "--host",
            "localhost",
            "--port",
            "9000",
        ]
    )

    async with AsyncClient(
        transport=ASGITransport(app=started["app"]),
        base_url="http://test",
    ) as client:
        response = await client.post("/v1/sessions")
    assert started["host"] == "localhost"
    assert started["port"] == 9000
    assert response.status_code == 201
    assert (workspace / ".yolop" / "runtime.db").is_file()
