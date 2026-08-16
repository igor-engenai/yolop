from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.usage import RunUsage
from yolop_session import (
    InvalidSessionIdError,
    SessionConflictError,
    SessionFormatError,
    SessionNotFoundError,
    SessionSnapshot,
    SessionStore,
)

from yolop import Yolop


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


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class _SessionLocks:
    def __init__(self) -> None:
        self._entries: dict[str, _LockEntry] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, session_id: str) -> AsyncIterator[None]:
        async with self._guard:
            entry = self._entries.setdefault(session_id, _LockEntry(lock=asyncio.Lock()))
            entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            async with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    del self._entries[session_id]


def create_app[DepsT](
    agent_spec: AgentSpec,
    session_store: SessionStore,
    *,
    deps: DepsT,
    deps_type: type[DepsT],
    model: Model | KnownModelName | str | None = None,
) -> FastAPI:
    """Create a FastAPI application for one host-selected AgentSpec."""
    app = FastAPI(title="YoloP")
    runtime = Yolop()
    locks = _SessionLocks()
    _install_session_error_handlers(app)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/sessions", response_model=SessionReference, status_code=201)
    async def create_session() -> SessionReference:
        return _session_reference(await session_store.create())

    @app.get("/v1/sessions", response_model=SessionListResponse)
    async def list_sessions() -> SessionListResponse:
        return SessionListResponse(sessions=await session_store.list_sessions())

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    async def load_session(session_id: str) -> SessionResponse:
        return _session_response(await session_store.load(session_id))

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> Response:
        async with locks.hold(session_id):
            session = await session_store.load(session_id)
            await session_store.delete(session_id, expected_revision=session.revision)
        return Response(status_code=204)

    @app.post("/v1/sessions/{session_id}/runs", response_model=RunResponse)
    async def run_agent(session_id: str, request: RunRequest) -> RunResponse:
        async with locks.hold(session_id):
            session = await session_store.load(session_id)
            async with runtime.run(
                agent_spec,
                request.prompt,
                deps=deps,
                deps_type=deps_type,
                model=model,
                message_history=session.messages,
            ) as run:
                async for _ in run:
                    pass
            assert run.result is not None
            saved = await session_store.replace(
                session_id,
                expected_revision=session.revision,
                messages=run.all_messages(),
            )
        return RunResponse(
            output=run.result.output,
            usage=run.result.usage,
            session=_session_reference(saved),
        )

    return app


def _session_reference(session: SessionSnapshot) -> SessionReference:
    return SessionReference(id=session.id, revision=session.revision)


def _session_response(session: SessionSnapshot) -> SessionResponse:
    messages = ModelMessagesTypeAdapter.dump_python(session.messages, mode="json")
    return SessionResponse(id=session.id, revision=session.revision, messages=messages)


def _install_session_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidSessionIdError)
    async def invalid_session_id(_request: Request, error: InvalidSessionIdError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found(_request: Request, error: SessionNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(SessionConflictError)
    async def session_conflict(_request: Request, error: SessionConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(SessionFormatError)
    async def invalid_session_data(_request: Request, _error: SessionFormatError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "Stored session data is invalid"})


__all__ = ["create_app"]
