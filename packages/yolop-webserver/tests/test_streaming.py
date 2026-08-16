import asyncio
import json
from collections.abc import AsyncIterator, MutableMapping
from typing import Any

from fastapi import Request
from httpx import ASGITransport, AsyncClient
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from yolop_session import ExecutionPin
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_webserver import create_app


def local_namespace(_request: Request) -> str:
    return "local"


def no_deps(_namespace: str, _session_id: str) -> None:
    return None


async def test_stream_returns_native_events_then_a_durable_completion(tmp_path) -> None:
    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "Hello "
        yield "stream"

    app = create_app(
        AgentSpec(),
        SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:function",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session = (await client.post("/v1/sessions")).json()
        async with client.stream(
            "POST",
            f"/v1/sessions/{session['id']}/runs/stream",
            json={"prompt": "Hello"},
        ) as response:
            content_type = response.headers["content-type"]
            body = (await response.aread()).decode()
        loaded = (await client.get(f"/v1/sessions/{session['id']}")).json()

    assert content_type.startswith("text/event-stream")
    events = _parse_sse(body)
    assert [name for name, _data in events] == [
        "part_start",
        "final_result",
        "part_delta",
        "part_end",
        "run_completed",
    ]
    assert json.loads(events[0][1])["event_kind"] == "part_start"
    completion = json.loads(events[-1][1])
    assert completion["output"] == "Hello stream"
    assert completion["session"] == {
        "id": session["id"],
        "revision": loaded["revision"],
    }
    assert [message["kind"] for message in loaded["messages"]] == ["request", "response"]


async def test_failed_stream_is_not_saved_and_returns_a_safe_error(tmp_path) -> None:
    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("secret provider detail")
        yield "unreachable"  # pragma: no cover

    app = create_app(
        AgentSpec(),
        SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=fail),
        model_id="test:function",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session = (await client.post("/v1/sessions")).json()
        response = await client.post(
            f"/v1/sessions/{session['id']}/runs/stream",
            json={"prompt": "Fail"},
        )
        loaded = (await client.get(f"/v1/sessions/{session['id']}")).json()

    assert response.status_code == 200
    assert _parse_sse(response.text) == [
        (
            "run_error",
            '{"code":"agent_run_failed","detail":"Agent run failed"}',
        )
    ]
    assert loaded["messages"] == []
    assert loaded["revision"] == session["revision"]
    assert "secret provider detail" not in response.text


async def test_disconnected_stream_is_cancelled_without_saving(tmp_path) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_forever(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        yield "unreachable"  # pragma: no cover

    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    spec = AgentSpec()
    session = await store.create_session(
        "local",
        pin=ExecutionPin.from_spec(spec, model_id="test:function"),
    )
    app = create_app(
        spec,
        store,
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=wait_forever),
        model_id="test:function",
    )
    request_sent = False

    async def receive() -> dict[str, object]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": json.dumps({"prompt": "Wait"}).encode(),
                "more_body": False,
            }
        await started.wait()
        return {"type": "http.disconnect"}

    async def send(_message: MutableMapping[str, Any]) -> None:
        pass

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": f"/v1/sessions/{session.id}/runs/stream",
            "raw_path": f"/v1/sessions/{session.id}/runs/stream".encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 80),
        },
        receive,
        send,
    )

    await asyncio.wait_for(cancelled.wait(), timeout=1)
    loaded = await store.load_session("local", session.id)
    assert loaded.messages == []
    assert loaded.revision == session.revision


async def test_different_sessions_can_stream_concurrently(tmp_path) -> None:
    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal active
        active += 1
        if active == 2:
            both_started.set()
        await release.wait()
        yield "done"

    app = create_app(
        AgentSpec(),
        SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:function",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        sessions = [(await client.post("/v1/sessions")).json()["id"] for _ in range(2)]
        requests = [
            asyncio.create_task(
                client.post(
                    f"/v1/sessions/{session_id}/runs/stream",
                    json={"prompt": "Run"},
                )
            )
            for session_id in sessions
        ]
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        responses = await asyncio.gather(*requests)

    assert all(response.status_code == 200 for response in responses)
    assert all(_parse_sse(response.text)[-1][0] == "run_completed" for response in responses)


def _parse_sse(body: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], fields["data"]))
    return events
