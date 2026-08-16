import asyncio
from collections.abc import AsyncIterator

from fastapi import Request
from httpx import ASGITransport, AsyncClient
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pytest import raises
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_webserver import RunLimits, create_app


def local_namespace(_request: Request) -> str:
    return "local"


def no_deps(_namespace: str, _session_id: str) -> None:
    return None


def test_model_object_requires_a_canonical_model_id(tmp_path) -> None:
    with raises(ValueError, match="model_id is required"):
        create_app(
            AgentSpec(),
            SQLiteRuntimeStore(tmp_path / "runtime.db"),
            namespace_resolver=local_namespace,
            deps_resolver=no_deps,
            deps_type=type(None),
            model=TestModel(),
        )


async def test_session_crud_is_isolated_by_the_resolved_namespace(tmp_path) -> None:
    def namespace(request: Request) -> str:
        return request.headers["x-tenant"]

    app = create_app(
        AgentSpec(),
        SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace_resolver=namespace,
        deps_resolver=lambda _namespace, _session_id: None,
        deps_type=type(None),
        model=TestModel(),
        model_id="test:model",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        acme = await client.post("/v1/sessions", headers={"x-tenant": "tenant/acme"})
        beta = await client.post("/v1/sessions", headers={"x-tenant": "tenant/beta"})
        acme_list = await client.get("/v1/sessions", headers={"x-tenant": "tenant/acme"})
        beta_list = await client.get("/v1/sessions", headers={"x-tenant": "tenant/beta"})
        cross_tenant_load = await client.get(
            f"/v1/sessions/{acme.json()['id']}",
            headers={"x-tenant": "tenant/beta"},
        )

    assert acme.status_code == 201
    assert beta.status_code == 201
    assert acme_list.json() == {"sessions": [acme.json()["id"]]}
    assert beta_list.json() == {"sessions": [beta.json()["id"]]}
    assert cross_tenant_load.status_code == 404
    assert cross_tenant_load.json()["code"] == "session_not_found"


async def test_session_rejects_a_different_agent_pin_before_model_execution(tmp_path) -> None:
    calls = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "must not run"

    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    first_app = create_app(
        AgentSpec(name="first"),
        store,
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:function",
    )
    second_app = create_app(
        AgentSpec(name="second"),
        store,
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:function",
    )

    async with AsyncClient(
        transport=ASGITransport(app=first_app), base_url="http://test"
    ) as client:
        session_id = (await client.post("/v1/sessions")).json()["id"]
    async with AsyncClient(
        transport=ASGITransport(app=second_app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/v1/sessions/{session_id}/runs",
            headers={"Idempotency-Key": "request-1"},
            json={"prompt": "Hello"},
        )

    assert response.status_code == 409
    assert response.json()["code"] == "session_pin_mismatch"
    assert calls == 0


async def test_client_can_manage_durable_sessions(tmp_path) -> None:
    app = create_app(
        AgentSpec(),
        SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=TestModel(),
        model_id="test:model",
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


async def test_json_run_requires_an_idempotency_key(tmp_path) -> None:
    calls = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "must not run"

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
        session_id = (await client.post("/v1/sessions")).json()["id"]
        response = await client.post(
            f"/v1/sessions/{session_id}/runs",
            json={"prompt": "Hello"},
        )

    assert response.status_code == 422
    assert calls == 0


async def test_json_run_uses_request_scoped_deps_and_persists_history(tmp_path) -> None:
    resolved: list[tuple[str, str]] = []
    calls = 0

    def resolve_deps(namespace: str, session_id: str) -> None:
        resolved.append((namespace, session_id))

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert prompt.content == "Hello"
        yield "Hello from YoloP"

    app = create_app(
        AgentSpec(),
        SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace_resolver=local_namespace,
        deps_resolver=resolve_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:function",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session = (await client.post("/v1/sessions")).json()
        response = await client.post(
            f"/v1/sessions/{session['id']}/runs",
            headers={"Idempotency-Key": "request-1"},
            json={"prompt": "Hello"},
        )
        retry = await client.post(
            f"/v1/sessions/{session['id']}/runs",
            headers={"Idempotency-Key": "request-1"},
            json={"prompt": "Hello"},
        )
        conflict = await client.post(
            f"/v1/sessions/{session['id']}/runs",
            headers={"Idempotency-Key": "request-1"},
            json={"prompt": "Different"},
        )
        loaded = (await client.get(f"/v1/sessions/{session['id']}")).json()

    assert response.status_code == 200
    assert retry.json() == response.json()
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"
    assert calls == 1
    assert resolved == [("local", session["id"])]
    result = response.json()
    assert result["output"] == "Hello from YoloP"
    assert result["session"] == {
        "id": session["id"],
        "revision": loaded["revision"],
    }
    assert result["usage"]["requests"] == 1
    assert [message["kind"] for message in loaded["messages"]] == ["request", "response"]


async def test_failed_json_run_is_replayed_without_a_second_model_call(tmp_path) -> None:
    calls = 0

    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider secret")
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
        session_id = (await client.post("/v1/sessions")).json()["id"]
        first = await client.post(
            f"/v1/sessions/{session_id}/runs",
            headers={"Idempotency-Key": "request-1"},
            json={"prompt": "Fail"},
        )
        retry = await client.post(
            f"/v1/sessions/{session_id}/runs",
            headers={"Idempotency-Key": "request-1"},
            json={"prompt": "Fail"},
        )

    assert first.status_code == 500
    assert (
        retry.json()
        == first.json()
        == {
            "code": "agent_run_failed",
            "detail": "Agent run failed",
        }
    )
    assert calls == 1


async def test_global_model_concurrency_is_bounded(tmp_path) -> None:
    active = 0
    maximum_active = 0
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            two_started.set()
        try:
            await release.wait()
            yield "done"
        finally:
            active -= 1

    app = create_app(
        AgentSpec(),
        SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:function",
        limits=RunLimits(max_active_runs=2, max_supervised_runs=3, poll_interval=0.005),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        sessions = [(await client.post("/v1/sessions")).json()["id"] for _ in range(3)]
        requests = [
            asyncio.create_task(
                client.post(
                    f"/v1/sessions/{session_id}/runs",
                    headers={"Idempotency-Key": f"request-{index}"},
                    json={"prompt": "Run"},
                )
            )
            for index, session_id in enumerate(sessions)
        ]
        await asyncio.wait_for(two_started.wait(), timeout=1)
        await asyncio.sleep(0.05)
        assert maximum_active == 2
        release.set()
        responses = await asyncio.gather(*requests)

    assert all(response.status_code == 200 for response in responses)
    assert maximum_active == 2


async def test_global_supervisor_rejects_excess_queued_runs(tmp_path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        started.set()
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
        limits=RunLimits(
            max_active_runs=1,
            max_supervised_runs=2,
            poll_interval=0.005,
        ),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        sessions = [(await client.post("/v1/sessions")).json()["id"] for _ in range(3)]
        first = asyncio.create_task(
            client.post(
                f"/v1/sessions/{sessions[0]}/runs",
                headers={"Idempotency-Key": "request-1"},
                json={"prompt": "First"},
            )
        )
        await started.wait()
        second = asyncio.create_task(
            client.post(
                f"/v1/sessions/{sessions[1]}/runs",
                headers={"Idempotency-Key": "request-2"},
                json={"prompt": "Second"},
            )
        )
        await asyncio.sleep(0.05)
        rejected = await client.post(
            f"/v1/sessions/{sessions[2]}/runs",
            headers={"Idempotency-Key": "request-3"},
            json={"prompt": "Third"},
        )
        release.set()
        completed = await asyncio.gather(first, second)

    assert rejected.status_code == 429
    assert rejected.json()["code"] == "run_queue_full"
    assert all(response.status_code == 200 for response in completed)


async def test_per_session_queue_rejects_excess_runs(tmp_path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        started.set()
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
        limits=RunLimits(max_pending_per_session=2, poll_interval=0.005),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session_id = (await client.post("/v1/sessions")).json()["id"]
        first = asyncio.create_task(
            client.post(
                f"/v1/sessions/{session_id}/runs",
                headers={"Idempotency-Key": "request-1"},
                json={"prompt": "First"},
            )
        )
        await started.wait()
        second = asyncio.create_task(
            client.post(
                f"/v1/sessions/{session_id}/runs",
                headers={"Idempotency-Key": "request-2"},
                json={"prompt": "Second"},
            )
        )
        await asyncio.sleep(0.05)
        rejected = await client.post(
            f"/v1/sessions/{session_id}/runs",
            headers={"Idempotency-Key": "request-3"},
            json={"prompt": "Third"},
        )
        release.set()
        completed = await asyncio.gather(first, second)

    assert rejected.status_code == 429
    assert rejected.json()["code"] == "run_queue_full"
    assert all(response.status_code == 200 for response in completed)


async def test_session_lock_timeout_returns_a_stable_service_error(tmp_path) -> None:
    calls = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "must not run"

    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    app = create_app(
        AgentSpec(),
        store,
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=respond),
        model_id="test:function",
        limits=RunLimits(session_lock_timeout=0.01, poll_interval=0.005),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        session_id = (await client.post("/v1/sessions")).json()["id"]
        async with store.lock_session("local", session_id, timeout=1):
            response = await client.post(
                f"/v1/sessions/{session_id}/runs",
                headers={"Idempotency-Key": "request-1"},
                json={"prompt": "Hello"},
            )

    assert response.status_code == 503
    assert response.json()["code"] == "session_lock_timeout"
    assert calls == 0


async def test_shutdown_interrupts_owned_runs_without_retrying_tools(tmp_path) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    async def wait_forever(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        yield "unreachable"  # pragma: no cover

    app = create_app(
        AgentSpec(),
        SQLiteRuntimeStore(tmp_path / "runtime.db"),
        namespace_resolver=local_namespace,
        deps_resolver=no_deps,
        deps_type=type(None),
        model=FunctionModel(stream_function=wait_forever),
        model_id="test:function",
        limits=RunLimits(poll_interval=0.005),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            session_id = (await client.post("/v1/sessions")).json()["id"]
            request = asyncio.create_task(
                client.post(
                    f"/v1/sessions/{session_id}/runs",
                    headers={"Idempotency-Key": "request-1"},
                    json={"prompt": "Wait"},
                )
            )
            await started.wait()
        response = await request
        retry = await client.post(
            f"/v1/sessions/{session_id}/runs",
            headers={"Idempotency-Key": "request-1"},
            json={"prompt": "Wait"},
        )

    assert response.status_code == 503
    assert retry.json() == response.json()
    assert response.json()["code"] == "run_interrupted"
    assert cancelled.is_set()
    assert calls == 1


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

    database = tmp_path / "runtime.db"
    model = FunctionModel(stream_function=respond)
    apps = [
        create_app(
            AgentSpec(),
            SQLiteRuntimeStore(database),
            namespace_resolver=local_namespace,
            deps_resolver=no_deps,
            deps_type=type(None),
            model=model,
            model_id="test:function",
        )
        for _ in range(2)
    ]

    async with (
        AsyncClient(transport=ASGITransport(app=apps[0]), base_url="http://test") as first_client,
        AsyncClient(transport=ASGITransport(app=apps[1]), base_url="http://test") as second_client,
    ):
        session_id = (await first_client.post("/v1/sessions")).json()["id"]
        first = asyncio.create_task(
            first_client.post(
                f"/v1/sessions/{session_id}/runs",
                headers={"Idempotency-Key": "request-1"},
                json={"prompt": "First"},
            )
        )
        await first_started.wait()
        second = asyncio.create_task(
            second_client.post(
                f"/v1/sessions/{session_id}/runs",
                headers={"Idempotency-Key": "request-2"},
                json={"prompt": "Second"},
            )
        )
        await asyncio.sleep(0)
        assert calls == 1
        release_first.set()
        responses = await asyncio.gather(first, second)
        loaded = (await first_client.get(f"/v1/sessions/{session_id}")).json()

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
