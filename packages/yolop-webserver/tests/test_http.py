import asyncio
from collections.abc import AsyncIterator

from httpx import ASGITransport, AsyncClient
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from yolop_sqlite_session import SQLiteSessionStore
from yolop_webserver import create_app


async def test_client_can_manage_durable_sessions(tmp_path) -> None:
    app = create_app(
        AgentSpec(),
        SQLiteSessionStore(tmp_path / "sessions.db"),
        deps=None,
        deps_type=type(None),
        model=TestModel(),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/healthz")
        created = await client.post("/v1/sessions")
        session = created.json()

        assert health.json() == {"status": "ok"}
        assert created.status_code == 201
        assert session["id"]
        assert session["revision"]
        listed = await client.get("/v1/sessions")
        assert listed.status_code == 200
        assert listed.json() == {"sessions": [session["id"]]}

        loaded = await client.get(f"/v1/sessions/{session['id']}")
        assert loaded.status_code == 200
        assert loaded.json() == {**session, "messages": []}

        assert (await client.delete(f"/v1/sessions/{session['id']}")).status_code == 204
        assert (await client.get(f"/v1/sessions/{session['id']}")).status_code == 404


async def test_json_run_uses_and_persists_the_session_history(tmp_path) -> None:
    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert prompt.content == "Hello"
        yield "Hello from YoloP"

    app = create_app(
        AgentSpec(),
        SQLiteSessionStore(tmp_path / "sessions.db"),
        deps=None,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session = (await client.post("/v1/sessions")).json()
        response = await client.post(
            f"/v1/sessions/{session['id']}/runs",
            json={"prompt": "Hello"},
        )
        loaded = (await client.get(f"/v1/sessions/{session['id']}")).json()

    assert response.status_code == 200
    result = response.json()
    assert result["output"] == "Hello from YoloP"
    assert result["session"] == {
        "id": session["id"],
        "revision": loaded["revision"],
    }
    assert result["usage"]["requests"] == 1
    assert [message["kind"] for message in loaded["messages"]] == ["request", "response"]


async def test_runs_for_one_session_wait_and_use_the_latest_history(tmp_path) -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls = 0

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
            yield "First answer"
        else:
            assert len(messages) == 3
            yield "Second answer"

    app = create_app(
        AgentSpec(),
        SQLiteSessionStore(tmp_path / "sessions.db"),
        deps=None,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session_id = (await client.post("/v1/sessions")).json()["id"]
        first = asyncio.create_task(
            client.post(f"/v1/sessions/{session_id}/runs", json={"prompt": "First"})
        )
        await first_started.wait()
        second = asyncio.create_task(
            client.post(f"/v1/sessions/{session_id}/runs", json={"prompt": "Second"})
        )
        await asyncio.sleep(0)
        assert calls == 1
        release_first.set()
        responses = await asyncio.gather(first, second)
        loaded = (await client.get(f"/v1/sessions/{session_id}")).json()

    assert [response.json()["output"] for response in responses] == [
        "First answer",
        "Second answer",
    ]
    assert [message["kind"] for message in loaded["messages"]] == [
        "request",
        "response",
        "request",
        "response",
    ]
