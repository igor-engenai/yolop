import json
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage


class InvalidNamespaceError(ValueError):
    """A runtime namespace is not a valid opaque host value."""

    code = "invalid_namespace"


class InvalidSessionIdError(ValueError):
    """A session ID is not a generated UUID4."""

    code = "invalid_session_id"


class SessionConflictError(RuntimeError):
    """The stored session does not match the expected revision."""

    code = "session_conflict"


class SessionFormatError(ValueError):
    """Stored session messages are not valid Pydantic AI messages."""

    code = "session_format_error"


class SessionNotFoundError(LookupError):
    """A session ID does not exist in the selected store."""

    code = "session_not_found"


class SessionPinMismatchError(RuntimeError):
    """A session is pinned to different agent execution configuration."""

    code = "session_pin_mismatch"


class IdempotencyConflictError(RuntimeError):
    """An idempotency key was reused for different input."""

    code = "idempotency_conflict"


class RunAdmissionError(RuntimeError):
    """A bounded run queue cannot accept more work."""

    code = "run_queue_full"


class RunNotFoundError(LookupError):
    """A run ID does not exist in the selected namespace."""

    code = "run_not_found"


class RunStateError(RuntimeError):
    """A run cannot make the requested state transition."""

    code = "run_state_conflict"


class RuntimeStoreSchemaError(RuntimeError):
    """Stored runtime data uses an unsupported schema."""

    code = "runtime_store_schema_mismatch"


class SessionLockTimeoutError(TimeoutError):
    """A session lock could not be acquired before its deadline."""

    code = "session_lock_timeout"


@dataclass(frozen=True)
class ExecutionPin:
    """Immutable agent and model identity for one session."""

    agent_spec_id: str
    model_id: str

    @classmethod
    def from_spec(
        cls,
        spec: AgentSpec | dict[str, Any],
        *,
        model_id: str,
    ) -> "ExecutionPin":
        return cls(agent_spec_id=agent_spec_digest(spec), model_id=model_id)


@dataclass(frozen=True)
class RuntimeSessionSnapshot:
    """A namespaced session at one content revision."""

    id: str
    namespace: str
    pin: ExecutionPin
    messages: list[ModelMessage]
    revision: str


class RunStatus(StrEnum):
    """Durable run lifecycle states."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class StoredRunEvent:
    """One ordered serialized event from a run."""

    sequence: int
    event: str
    data: str


@dataclass(frozen=True)
class RuntimeRunSnapshot:
    """Durable run state inside one namespace and session."""

    id: str
    namespace: str
    session_id: str
    idempotency_key: str
    prompt: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    owner_id: str | None = None
    lease_expires_at: datetime | None = None
    output: Any | None = None
    usage: RunUsage | None = None
    session_revision: str | None = None
    error_code: str | None = None
    error_detail: str | None = None


@dataclass(frozen=True)
class RunReservation:
    """Result of idempotent run submission."""

    run: RuntimeRunSnapshot
    created: bool


@dataclass(frozen=True)
class RunCompletion:
    """One atomic session and run completion."""

    session: RuntimeSessionSnapshot
    run: RuntimeRunSnapshot


class RuntimeStore(Protocol):
    """Namespaced durable state and coordination required by YoloP hosts."""

    async def create_session(
        self,
        namespace: str,
        *,
        pin: ExecutionPin,
    ) -> RuntimeSessionSnapshot: ...

    async def list_sessions(self, namespace: str) -> list[str]: ...

    async def load_session(self, namespace: str, session_id: str) -> RuntimeSessionSnapshot: ...

    async def delete_session(
        self,
        namespace: str,
        session_id: str,
        *,
        expected_revision: str,
    ) -> None: ...

    async def replace_session(
        self,
        namespace: str,
        session_id: str,
        *,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> RuntimeSessionSnapshot: ...

    def lock_session(
        self,
        namespace: str,
        session_id: str,
        *,
        timeout: float,
    ) -> AbstractAsyncContextManager[None]: ...

    async def reserve_run(
        self,
        namespace: str,
        session_id: str,
        *,
        idempotency_key: str,
        prompt: str,
        max_pending: int | None = None,
    ) -> RunReservation: ...

    async def load_run(self, namespace: str, run_id: str) -> RuntimeRunSnapshot: ...

    async def claim_run(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot: ...

    async def renew_run_lease(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: float,
    ) -> RuntimeRunSnapshot: ...

    async def append_run_event(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        event: str,
        data: str,
    ) -> StoredRunEvent: ...

    async def list_run_events(
        self,
        namespace: str,
        run_id: str,
        *,
        after: int = 0,
    ) -> list[StoredRunEvent]: ...

    async def complete_run(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        expected_session_revision: str,
        messages: Sequence[ModelMessage],
        output: Any,
        usage: RunUsage,
    ) -> RunCompletion: ...

    async def fail_run(
        self,
        namespace: str,
        run_id: str,
        *,
        owner_id: str,
        error_code: str,
        error_detail: str,
    ) -> RuntimeRunSnapshot: ...

    async def interrupt_owned_runs(self, owner_id: str) -> int: ...

    async def interrupt_expired_runs(self) -> int: ...


def ensure_session_pin(session: RuntimeSessionSnapshot, expected: ExecutionPin) -> None:
    """Reject execution through configuration other than the session pin."""
    if session.pin != expected:
        raise SessionPinMismatchError(
            f"Session {session.id!r} is pinned to different agent configuration"
        )


def agent_spec_digest(spec: AgentSpec | dict[str, Any]) -> str:
    """Return a stable digest for canonical AgentSpec data."""
    validated = spec if isinstance(spec, AgentSpec) else AgentSpec.model_validate(spec)
    canonical = json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(canonical).hexdigest()


def new_session_id() -> str:
    """Return a canonical generated UUID4 session ID."""
    return str(uuid4())


def validate_namespace(namespace: str) -> str:
    """Return a bounded opaque namespace or raise."""
    if not isinstance(namespace, str) or not namespace.strip() or len(namespace) > 255:
        raise InvalidNamespaceError("Namespace must contain between 1 and 255 characters")
    return namespace


def validate_session_id(session_id: str) -> str:
    """Return a canonical generated UUID4 session ID or raise."""
    try:
        parsed = UUID(session_id)
    except (AttributeError, TypeError, ValueError) as error:
        raise InvalidSessionIdError(
            f"Session ID {session_id!r} is not a generated UUID4"
        ) from error
    if parsed.version != 4 or str(parsed) != session_id:
        raise InvalidSessionIdError(f"Session ID {session_id!r} is not a generated UUID4")
    return session_id


__all__ = [
    "ExecutionPin",
    "IdempotencyConflictError",
    "InvalidNamespaceError",
    "InvalidSessionIdError",
    "RunAdmissionError",
    "RunCompletion",
    "RunNotFoundError",
    "RunReservation",
    "RunStateError",
    "RunStatus",
    "RuntimeRunSnapshot",
    "RuntimeSessionSnapshot",
    "RuntimeStore",
    "RuntimeStoreSchemaError",
    "SessionConflictError",
    "SessionFormatError",
    "SessionLockTimeoutError",
    "SessionNotFoundError",
    "SessionPinMismatchError",
    "StoredRunEvent",
    "agent_spec_digest",
    "ensure_session_pin",
    "new_session_id",
    "validate_namespace",
    "validate_session_id",
]
