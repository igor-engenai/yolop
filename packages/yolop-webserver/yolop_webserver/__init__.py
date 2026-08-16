from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import AgentRunResultEvent, AgentSpec, AgentStreamEvent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.usage import RunUsage
from sse_starlette import EventSourceResponse
from yolop_session import (
    ExecutionPin,
    InvalidSessionIdError,
    RuntimeSessionSnapshot,
    RuntimeStore,
    SessionConflictError,
    SessionFormatError,
    SessionLockTimeoutError,
    SessionNotFoundError,
    SessionPinMismatchError,
    ensure_session_pin,
    validate_namespace,
)

from yolop import Yolop

_LOGGER = logging.getLogger(__name__)
_STREAM_EVENT_ADAPTER = TypeAdapter(AgentStreamEvent)

type NamespaceResolver = Callable[[Request], str | Awaitable[str]]
type DepsResolver[DepsT] = Callable[[str, str], DepsT | Awaitable[DepsT]]


class SessionReference(BaseModel):
    id: str
    revision: str


class SessionResponse(SessionReference):
    messages: list[dict[str, Any]]


class SessionListResponse(BaseModel):
    sessions: list[str]


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1)


class RunResponse(BaseModel):
    output: Any
    usage: RunUsage
    session: SessionReference


def create_app[DepsT](
    agent_spec: AgentSpec,
    runtime_store: RuntimeStore,
    *,
    namespace_resolver: NamespaceResolver,
    deps_resolver: DepsResolver[DepsT],
    deps_type: type[DepsT],
    model: Model | KnownModelName | str | None = None,
    model_id: str | None = None,
) -> FastAPI:
    """Create a FastAPI application for one host-selected AgentSpec."""
    app = FastAPI(title="YoloP")
    runtime = Yolop()
    pin = _execution_pin(agent_spec, model=model, model_id=model_id)
    _install_runtime_error_handlers(app)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/sessions", response_model=SessionReference, status_code=201)
    async def create_session(request: Request) -> SessionReference:
        namespace = await _resolve_namespace(namespace_resolver, request)
        return _session_reference(await runtime_store.create_session(namespace, pin=pin))

    @app.get("/v1/sessions", response_model=SessionListResponse)
    async def list_sessions(request: Request) -> SessionListResponse:
        namespace = await _resolve_namespace(namespace_resolver, request)
        return SessionListResponse(sessions=await runtime_store.list_sessions(namespace))

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    async def load_session(request: Request, session_id: str) -> SessionResponse:
        namespace = await _resolve_namespace(namespace_resolver, request)
        return _session_response(await runtime_store.load_session(namespace, session_id))

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    async def delete_session(request: Request, session_id: str) -> Response:
        namespace = await _resolve_namespace(namespace_resolver, request)
        async with runtime_store.lock_session(namespace, session_id, timeout=30):
            session = await runtime_store.load_session(namespace, session_id)
            await runtime_store.delete_session(
                namespace,
                session_id,
                expected_revision=session.revision,
            )
        return Response(status_code=204)

    @app.post("/v1/sessions/{session_id}/runs", response_model=RunResponse)
    async def run_agent(
        request: Request,
        session_id: str,
        run_request: RunRequest,
    ) -> RunResponse:
        namespace = await _resolve_namespace(namespace_resolver, request)
        async with runtime_store.lock_session(namespace, session_id, timeout=30):
            session = await runtime_store.load_session(namespace, session_id)
            ensure_session_pin(session, pin)
            deps = await _resolve_deps(deps_resolver, namespace, session_id)
            async with runtime.run(
                agent_spec,
                run_request.prompt,
                deps=deps,
                deps_type=deps_type,
                model=model,
                message_history=session.messages,
            ) as run:
                async for _ in run:
                    pass
            assert run.result is not None
            saved = await runtime_store.replace_session(
                namespace,
                session_id,
                expected_revision=session.revision,
                messages=run.all_messages(),
            )
        return RunResponse(
            output=run.result.output,
            usage=run.result.usage,
            session=_session_reference(saved),
        )

    @app.post("/v1/sessions/{session_id}/runs/stream")
    async def stream_agent(
        request: Request,
        session_id: str,
        run_request: RunRequest,
    ) -> EventSourceResponse:
        namespace = await _resolve_namespace(namespace_resolver, request)
        initial_session = await runtime_store.load_session(namespace, session_id)
        ensure_session_pin(initial_session, pin)

        async def events() -> AsyncIterator[dict[str, str]]:
            try:
                async with runtime_store.lock_session(namespace, session_id, timeout=30):
                    session = await runtime_store.load_session(namespace, session_id)
                    ensure_session_pin(session, pin)
                    deps = await _resolve_deps(deps_resolver, namespace, session_id)
                    async with runtime.run(
                        agent_spec,
                        run_request.prompt,
                        deps=deps,
                        deps_type=deps_type,
                        model=model,
                        message_history=session.messages,
                    ) as run:
                        async for event in run:
                            if isinstance(event, AgentRunResultEvent):
                                continue
                            yield {
                                "event": event.event_kind,
                                "data": _STREAM_EVENT_ADAPTER.dump_json(event).decode(),
                            }
                    assert run.result is not None
                    saved = await runtime_store.replace_session(
                        namespace,
                        session_id,
                        expected_revision=session.revision,
                        messages=run.all_messages(),
                    )
                completion = RunResponse(
                    output=run.result.output,
                    usage=run.result.usage,
                    session=_session_reference(saved),
                )
                yield {"event": "run_completed", "data": completion.model_dump_json()}
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("YoloP streaming run failed for session %s", session_id)
                yield {
                    "event": "run_error",
                    "data": '{"code":"agent_run_failed","detail":"Agent run failed"}',
                }

        return EventSourceResponse(events())

    return app


def _execution_pin(
    agent_spec: AgentSpec,
    *,
    model: Model | KnownModelName | str | None,
    model_id: str | None,
) -> ExecutionPin:
    resolved_id = model_id
    if resolved_id is None and isinstance(model, str):
        resolved_id = model
    if resolved_id is None and model is None and isinstance(agent_spec.model, str):
        resolved_id = agent_spec.model
    if not resolved_id:
        raise ValueError("model_id is required when the resolved model is not a string reference")
    return ExecutionPin.from_spec(agent_spec, model_id=resolved_id)


async def _resolve_namespace(resolver: NamespaceResolver, request: Request) -> str:
    value = resolver(request)
    if inspect.isawaitable(value):
        return validate_namespace(cast(str, await value))
    return validate_namespace(cast(str, value))


async def _resolve_deps[DepsT](
    resolver: DepsResolver[DepsT],
    namespace: str,
    session_id: str,
) -> DepsT:
    value = resolver(namespace, session_id)
    if inspect.isawaitable(value):
        return cast(DepsT, await value)
    return cast(DepsT, value)


def _session_reference(session: RuntimeSessionSnapshot) -> SessionReference:
    return SessionReference(id=session.id, revision=session.revision)


def _session_response(session: RuntimeSessionSnapshot) -> SessionResponse:
    messages = ModelMessagesTypeAdapter.dump_python(session.messages, mode="json")
    return SessionResponse(id=session.id, revision=session.revision, messages=messages)


def _install_runtime_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidSessionIdError)
    async def invalid_session_id(_request: Request, error: InvalidSessionIdError) -> JSONResponse:
        return _error_response(422, error.code, str(error))

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found(_request: Request, error: SessionNotFoundError) -> JSONResponse:
        return _error_response(404, error.code, str(error))

    @app.exception_handler(SessionConflictError)
    async def session_conflict(_request: Request, error: SessionConflictError) -> JSONResponse:
        return _error_response(409, error.code, str(error))

    @app.exception_handler(SessionPinMismatchError)
    async def session_pin_mismatch(
        _request: Request,
        error: SessionPinMismatchError,
    ) -> JSONResponse:
        return _error_response(409, error.code, str(error))

    @app.exception_handler(SessionLockTimeoutError)
    async def session_lock_timeout(
        _request: Request,
        error: SessionLockTimeoutError,
    ) -> JSONResponse:
        return _error_response(503, error.code, str(error))

    @app.exception_handler(SessionFormatError)
    async def invalid_session_data(_request: Request, error: SessionFormatError) -> JSONResponse:
        return _error_response(500, error.code, "Stored session data is invalid")


def _error_response(status_code: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"code": code, "detail": detail})


__all__ = ["DepsResolver", "NamespaceResolver", "create_app"]
