from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
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
    RunAdmissionError,
    RunNotFoundError,
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


@dataclass(frozen=True)
class RunLimits:
    """Bound model execution and queued HTTP work."""

    max_active_runs: int = 8
    max_pending_per_session: int = 8
    max_supervised_runs: int = 64
    session_lock_timeout: float = 30
    lease_seconds: float = 60
    poll_interval: float = 0.02

    def __post_init__(self) -> None:
        values = (
            self.max_active_runs,
            self.max_pending_per_session,
            self.max_supervised_runs,
            self.session_lock_timeout,
            self.lease_seconds,
            self.poll_interval,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Run limits must be positive")


type _RunExecutor = Callable[
    [RuntimeRunSnapshot, asyncio.Semaphore],
    Coroutine[Any, Any, None],
]


class _RunSupervisor:
    def __init__(
        self,
        runtime_store: RuntimeStore,
        limits: RunLimits,
        executor: _RunExecutor,
    ) -> None:
        self.owner_id = str(uuid4())
        self._runtime_store = runtime_store
        self._limits = limits
        self._executor = executor
        self._active = asyncio.Semaphore(limits.max_active_runs)
        self._guard = asyncio.Lock()
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}

    async def start(self, namespace: str, run_id: str) -> None:
        key = (namespace, run_id)
        async with self._guard:
            if key in self._tasks:
                return
            run = await self._runtime_store.load_run(namespace, run_id)
            if run.status is not RunStatus.ACCEPTED:
                return
            if len(self._tasks) >= self._limits.max_supervised_runs:
                try:
                    claimed = await self._runtime_store.claim_run(
                        namespace,
                        run_id,
                        owner_id=self.owner_id,
                        lease_seconds=self._limits.lease_seconds,
                    )
                except RunStateError:
                    return
                await self._runtime_store.fail_run(
                    namespace,
                    claimed.id,
                    owner_id=self.owner_id,
                    error_code=RunAdmissionError.code,
                    error_detail="Run supervisor is at capacity",
                )
                raise RunAdmissionError("Run supervisor is at capacity")
            try:
                claimed = await self._runtime_store.claim_run(
                    namespace,
                    run_id,
                    owner_id=self.owner_id,
                    lease_seconds=self._limits.lease_seconds,
                )
            except RunStateError:
                return
            task = asyncio.create_task(
                self._executor(claimed, self._active),
                name=f"yolop-run-{run_id}",
            )
            self._tasks[key] = task
            task.add_done_callback(lambda completed: self._task_done(key, completed))

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._runtime_store.interrupt_owned_runs(self.owner_id)

    def _task_done(self, key: tuple[str, str], task: asyncio.Task[None]) -> None:
        self._tasks.pop(key, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            _LOGGER.error(
                "Unobserved YoloP background run failure",
                exc_info=(type(error), error, error.__traceback__),
            )


def create_app[DepsT](
    agent_spec: AgentSpec,
    runtime_store: RuntimeStore,
    *,
    namespace_resolver: NamespaceResolver,
    deps_resolver: DepsResolver[DepsT],
    deps_type: type[DepsT],
    model: Model | KnownModelName | str | None = None,
    model_id: str | None = None,
    limits: RunLimits | None = None,
) -> FastAPI:
    """Create a FastAPI application for one host-selected AgentSpec."""
    runtime = Yolop()
    pin = _execution_pin(agent_spec, model=model, model_id=model_id)
    configured_limits = limits or RunLimits()

    async def execute_run(
        claimed: RuntimeRunSnapshot,
        active_runs: asyncio.Semaphore,
    ) -> None:
        assert claimed.owner_id is not None
        heartbeat = asyncio.create_task(
            _heartbeat_run(
                runtime_store,
                claimed,
                lease_seconds=configured_limits.lease_seconds,
            )
        )
        try:
            async with runtime_store.lock_session(
                claimed.namespace,
                claimed.session_id,
                timeout=configured_limits.session_lock_timeout,
            ):
                async with active_runs:
                    session = await runtime_store.load_session(
                        claimed.namespace,
                        claimed.session_id,
                    )
                    ensure_session_pin(session, pin)
                    deps = await _resolve_deps(
                        deps_resolver,
                        claimed.namespace,
                        claimed.session_id,
                    )
                    async with runtime.run(
                        agent_spec,
                        claimed.prompt,
                        deps=deps,
                        deps_type=deps_type,
                        model=model,
                        message_history=session.messages,
                    ) as agent_run:
                        async for event in agent_run:
                            if isinstance(event, AgentRunResultEvent):
                                continue
                            await runtime_store.append_run_event(
                                claimed.namespace,
                                claimed.id,
                                owner_id=claimed.owner_id,
                                event=event.event_kind,
                                data=_STREAM_EVENT_ADAPTER.dump_json(event).decode(),
                            )
                    assert agent_run.result is not None
                    await runtime_store.complete_run(
                        claimed.namespace,
                        claimed.id,
                        owner_id=claimed.owner_id,
                        expected_session_revision=session.revision,
                        messages=agent_run.all_messages(),
                        output=agent_run.result.output,
                        usage=agent_run.result.usage,
                    )
        except asyncio.CancelledError:
            raise
        except SessionLockTimeoutError as error:
            await _fail_owned_run(
                runtime_store,
                claimed,
                error_code=error.code,
                error_detail=str(error),
            )
        except Exception:
            await _fail_owned_run(
                runtime_store,
                claimed,
                error_code="agent_run_failed",
                error_detail="Agent run failed",
            )
            _LOGGER.exception("YoloP background run failed for session %s", claimed.session_id)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    supervisor = _RunSupervisor(runtime_store, configured_limits, execute_run)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await runtime_store.interrupt_expired_runs()
        try:
            yield
        finally:
            await supervisor.shutdown()

    app = FastAPI(title="YoloP", lifespan=lifespan)
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
        async with runtime_store.lock_session(
            namespace,
            session_id,
            timeout=configured_limits.session_lock_timeout,
        ):
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
        await runtime_store.interrupt_expired_runs()
        initial_session = await runtime_store.load_session(namespace, session_id)
        ensure_session_pin(initial_session, pin)
        reservation = await runtime_store.reserve_run(
            namespace,
            session_id,
            idempotency_key=idempotency_key,
            prompt=run_request.prompt,
            max_pending=configured_limits.max_pending_per_session,
        )
        await supervisor.start(namespace, reservation.run.id)
        terminal = await _wait_for_terminal_run(
            runtime_store,
            namespace,
            reservation.run.id,
            poll_interval=configured_limits.poll_interval,
        )
        if terminal.status is RunStatus.COMPLETED:
            return _completed_run_response(terminal)
        raise _terminal_run_error(terminal)

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
        await runtime_store.interrupt_expired_runs()
        initial_session = await runtime_store.load_session(namespace, session_id)
        ensure_session_pin(initial_session, pin)
        reservation = await runtime_store.reserve_run(
            namespace,
            session_id,
            idempotency_key=idempotency_key,
            prompt=run_request.prompt,
            max_pending=configured_limits.max_pending_per_session,
        )
        await supervisor.start(namespace, reservation.run.id)

        async def events() -> AsyncIterator[dict[str, str]]:
            async for event in _follow_run_events(
                runtime_store,
                namespace,
                reservation.run.id,
                poll_interval=configured_limits.poll_interval,
            ):
                yield event

        return EventSourceResponse(events())

    return app


async def _heartbeat_run(
    runtime_store: RuntimeStore,
    run: RuntimeRunSnapshot,
    *,
    lease_seconds: float,
) -> None:
    assert run.owner_id is not None
    interval = lease_seconds / 3
    while True:
        await asyncio.sleep(interval)
        await runtime_store.renew_run_lease(
            run.namespace,
            run.id,
            owner_id=run.owner_id,
            lease_seconds=lease_seconds,
        )


async def _fail_owned_run(
    runtime_store: RuntimeStore,
    run: RuntimeRunSnapshot,
    *,
    error_code: str,
    error_detail: str,
) -> None:
    assert run.owner_id is not None
    await runtime_store.fail_run(
        run.namespace,
        run.id,
        owner_id=run.owner_id,
        error_code=error_code,
        error_detail=error_detail,
    )


async def _wait_for_terminal_run(
    runtime_store: RuntimeStore,
    namespace: str,
    run_id: str,
    *,
    poll_interval: float,
) -> RuntimeRunSnapshot:
    while True:
        run = await runtime_store.load_run(namespace, run_id)
        if _run_lease_expired(run):
            await runtime_store.interrupt_expired_runs()
            continue
        if run.status not in {RunStatus.ACCEPTED, RunStatus.RUNNING}:
            return run
        await asyncio.sleep(poll_interval)


async def _follow_run_events(
    runtime_store: RuntimeStore,
    namespace: str,
    run_id: str,
    *,
    poll_interval: float,
) -> AsyncIterator[dict[str, str]]:
    sequence = 0
    while True:
        stored_events = await runtime_store.list_run_events(
            namespace,
            run_id,
            after=sequence,
        )
        for stored_event in stored_events:
            sequence = stored_event.sequence
            yield _sse_event(stored_event)

        run = await runtime_store.load_run(namespace, run_id)
        if _run_lease_expired(run):
            await runtime_store.interrupt_expired_runs()
            continue
        if run.status not in {RunStatus.ACCEPTED, RunStatus.RUNNING}:
            final_events = await runtime_store.list_run_events(
                namespace,
                run_id,
                after=sequence,
            )
            for stored_event in final_events:
                sequence = stored_event.sequence
                yield _sse_event(stored_event)
            if run.status is RunStatus.COMPLETED:
                yield _completion_event(run)
            else:
                yield _run_error_event(run)
            return
        await asyncio.sleep(poll_interval)


def _run_lease_expired(run: RuntimeRunSnapshot) -> bool:
    return (
        run.status is RunStatus.RUNNING
        and run.lease_expires_at is not None
        and run.lease_expires_at <= datetime.now(UTC)
    )


def _terminal_run_error(run: RuntimeRunSnapshot) -> _TerminalRunError:
    if run.error_code == RunAdmissionError.code:
        status_code = 429
    elif run.status is RunStatus.INTERRUPTED or run.error_code == SessionLockTimeoutError.code:
        status_code = 503
    else:
        status_code = 500
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

    @app.exception_handler(RunNotFoundError)
    async def run_not_found(_request: Request, error: RunNotFoundError) -> JSONResponse:
        return _error_response(404, error.code, str(error))

    @app.exception_handler(RunAdmissionError)
    async def run_queue_full(_request: Request, error: RunAdmissionError) -> JSONResponse:
        return _error_response(429, error.code, str(error))

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


__all__ = ["DepsResolver", "NamespaceResolver", "RunLimits", "create_app"]
