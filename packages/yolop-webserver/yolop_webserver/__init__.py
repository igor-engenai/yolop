from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import AgentRunResultEvent, AgentSpec, AgentStreamEvent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.usage import RunUsage
from sse_starlette import EventSourceResponse
from yolop_session import (
    ExecutionPin,
    IdempotencyConflictError,
    InvalidSessionIdError,
    RunStateError,
    RunStatus,
    RuntimeRunSnapshot,
    RuntimeSessionSnapshot,
    RuntimeStore,
    SessionConflictError,
    SessionFormatError,
    SessionLockTimeoutError,
    SessionNotFoundError,
    SessionPinMismatchError,
    StoredRunEvent,
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
    run_id: str
    output: Any
    usage: RunUsage
    session: SessionReference


class _TerminalRunError(RuntimeError):
    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail


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
    owner_id = str(uuid4())
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
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1, max_length=255),
        ],
    ) -> RunResponse:
        namespace = await _resolve_namespace(namespace_resolver, request)
        initial_session = await runtime_store.load_session(namespace, session_id)
        ensure_session_pin(initial_session, pin)
        reservation = await runtime_store.reserve_run(
            namespace,
            session_id,
            idempotency_key=idempotency_key,
            prompt=run_request.prompt,
        )
        async with runtime_store.lock_session(namespace, session_id, timeout=30):
            stored_run = await runtime_store.load_run(namespace, reservation.run.id)
            if stored_run.status is RunStatus.COMPLETED:
                return _completed_run_response(stored_run)
            if stored_run.status in {RunStatus.FAILED, RunStatus.INTERRUPTED}:
                raise _terminal_run_error(stored_run)
            if stored_run.status is not RunStatus.ACCEPTED:
                raise RunStateError(f"Run {stored_run.id!r} is already active")
            stored_run = await runtime_store.claim_run(
                namespace,
                stored_run.id,
                owner_id=owner_id,
                lease_seconds=60,
            )
            session = await runtime_store.load_session(namespace, session_id)
            ensure_session_pin(session, pin)
            deps = await _resolve_deps(deps_resolver, namespace, session_id)
            try:
                async with runtime.run(
                    agent_spec,
                    run_request.prompt,
                    deps=deps,
                    deps_type=deps_type,
                    model=model,
                    message_history=session.messages,
                ) as agent_run:
                    async for event in agent_run:
                        if isinstance(event, AgentRunResultEvent):
                            continue
                        await runtime_store.append_run_event(
                            namespace,
                            stored_run.id,
                            owner_id=owner_id,
                            event=event.event_kind,
                            data=_STREAM_EVENT_ADAPTER.dump_json(event).decode(),
                        )
                assert agent_run.result is not None
                completion = await runtime_store.complete_run(
                    namespace,
                    stored_run.id,
                    owner_id=owner_id,
                    expected_session_revision=session.revision,
                    messages=agent_run.all_messages(),
                    output=agent_run.result.output,
                    usage=agent_run.result.usage,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failed = await runtime_store.fail_run(
                    namespace,
                    stored_run.id,
                    owner_id=owner_id,
                    error_code="agent_run_failed",
                    error_detail="Agent run failed",
                )
                _LOGGER.exception("YoloP run failed for session %s", session_id)
                raise _terminal_run_error(failed) from error
        return _completed_run_response(completion.run)

    @app.post("/v1/sessions/{session_id}/runs/stream")
    async def stream_agent(
        request: Request,
        session_id: str,
        run_request: RunRequest,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1, max_length=255),
        ],
    ) -> EventSourceResponse:
        namespace = await _resolve_namespace(namespace_resolver, request)
        initial_session = await runtime_store.load_session(namespace, session_id)
        ensure_session_pin(initial_session, pin)
        reservation = await runtime_store.reserve_run(
            namespace,
            session_id,
            idempotency_key=idempotency_key,
            prompt=run_request.prompt,
        )

        async def events() -> AsyncIterator[dict[str, str]]:
            claimed = False
            try:
                async with runtime_store.lock_session(namespace, session_id, timeout=30):
                    stored_run = await runtime_store.load_run(namespace, reservation.run.id)
                    if stored_run.status is not RunStatus.ACCEPTED:
                        for stored_event in await runtime_store.list_run_events(
                            namespace,
                            stored_run.id,
                        ):
                            yield _sse_event(stored_event)
                        if stored_run.status is RunStatus.COMPLETED:
                            yield _completion_event(stored_run)
                        else:
                            yield _run_error_event(stored_run)
                        return

                    stored_run = await runtime_store.claim_run(
                        namespace,
                        stored_run.id,
                        owner_id=owner_id,
                        lease_seconds=60,
                    )
                    claimed = True
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
                    ) as agent_run:
                        async for event in agent_run:
                            if isinstance(event, AgentRunResultEvent):
                                continue
                            stored_event = await runtime_store.append_run_event(
                                namespace,
                                stored_run.id,
                                owner_id=owner_id,
                                event=event.event_kind,
                                data=_STREAM_EVENT_ADAPTER.dump_json(event).decode(),
                            )
                            yield _sse_event(stored_event)
                    assert agent_run.result is not None
                    completion = await runtime_store.complete_run(
                        namespace,
                        stored_run.id,
                        owner_id=owner_id,
                        expected_session_revision=session.revision,
                        messages=agent_run.all_messages(),
                        output=agent_run.result.output,
                        usage=agent_run.result.usage,
                    )
                yield _completion_event(completion.run)
            except asyncio.CancelledError:
                raise
            except Exception:
                if claimed:
                    await runtime_store.fail_run(
                        namespace,
                        reservation.run.id,
                        owner_id=owner_id,
                        error_code="agent_run_failed",
                        error_detail="Agent run failed",
                    )
                _LOGGER.exception("YoloP streaming run failed for session %s", session_id)
                yield {
                    "event": "run_error",
                    "data": '{"code":"agent_run_failed","detail":"Agent run failed"}',
                }

        return EventSourceResponse(events())

    return app


def _terminal_run_error(run: RuntimeRunSnapshot) -> _TerminalRunError:
    status_code = 503 if run.status is RunStatus.INTERRUPTED else 500
    return _TerminalRunError(
        status_code,
        run.error_code or "run_failed",
        run.error_detail or "Run failed",
    )


def _sse_event(event: StoredRunEvent) -> dict[str, str]:
    return {"id": str(event.sequence), "event": event.event, "data": event.data}


def _completion_event(run: RuntimeRunSnapshot) -> dict[str, str]:
    return {
        "event": "run_completed",
        "data": _completed_run_response(run).model_dump_json(),
    }


def _run_error_event(run: RuntimeRunSnapshot) -> dict[str, str]:
    code = run.error_code or "run_state_conflict"
    detail = run.error_detail or "Run did not complete"
    return {
        "event": "run_error",
        "data": json.dumps({"code": code, "detail": detail}, separators=(",", ":")),
    }


def _completed_run_response(run: RuntimeRunSnapshot) -> RunResponse:
    if (
        run.status is not RunStatus.COMPLETED
        or run.usage is None
        or run.session_revision is None
    ):
        raise RunStateError(f"Run {run.id!r} has no durable completion")
    return RunResponse(
        run_id=run.id,
        output=run.output,
        usage=run.usage,
        session=SessionReference(id=run.session_id, revision=run.session_revision),
    )


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

    @app.exception_handler(_TerminalRunError)
    async def terminal_run_error(_request: Request, error: _TerminalRunError) -> JSONResponse:
        return _error_response(error.status_code, error.code, error.detail)

    @app.exception_handler(IdempotencyConflictError)
    async def idempotency_conflict(
        _request: Request,
        error: IdempotencyConflictError,
    ) -> JSONResponse:
        return _error_response(409, error.code, str(error))

    @app.exception_handler(RunStateError)
    async def run_state_conflict(_request: Request, error: RunStateError) -> JSONResponse:
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
