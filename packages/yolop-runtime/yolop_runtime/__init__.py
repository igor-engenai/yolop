import json
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol, TypeVar
from uuid import UUID, uuid4

from pydantic_ai import AgentSpec, AgentStreamEvent
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
class ExecutionScope:
    """Identity for one durable runtime execution."""

    namespace: str
    session_id: str
    run_id: str
    parent_run_id: str | None = None
    root_run_id: str | None = None
    initiator: str = "user"

    def __post_init__(self) -> None:
        validate_namespace(self.namespace)
        validate_session_id(self.session_id)
        validate_session_id(self.run_id)
        if self.parent_run_id is not None:
            validate_session_id(self.parent_run_id)
        if self.root_run_id is None:
            object.__setattr__(self, "root_run_id", self.run_id)
        else:
            validate_session_id(self.root_run_id)
        if not self.initiator.strip():
            raise ValueError("Execution initiator must not be empty")


class ScopedState(Protocol):
    """Host-provided state access already scoped to the current execution."""

    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: Any) -> None: ...

    async def delete(self, key: str) -> None: ...


class RuntimeEventSink(Protocol):
    """Receives native Pydantic AI stream events without translation."""

    async def emit(self, event: AgentStreamEvent) -> None: ...


class RuntimeFollowUpSink(Protocol):
    """Receives host follow-up prompts for the current execution."""

    async def enqueue(self, prompt: str) -> None: ...


HostDepsT = TypeVar("HostDepsT")
StateT = TypeVar("StateT")


@dataclass(frozen=True)
class RuntimeDeps[HostDepsT, StateT]:
    """Dependencies exposed to one runtime execution."""

    scope: ExecutionScope
    state: StateT | None
    event_sink: RuntimeEventSink | None
    follow_up_sink: RuntimeFollowUpSink | None
    host: HostDepsT

    def __getattr__(self, name: str) -> Any:
        """Keep existing host capabilities usable while exposing the runtime envelope."""
        return getattr(self.host, name)


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

    async def list_runs(
        self,
        namespace: str,
        *,
        session_id: str | None = None,
    ) -> list[RuntimeRunSnapshot]: ...

    async def cancel_run(
        self,
        namespace: str,
        run_id: str,
        *,
        error_code: str = "run_cancelled",
        error_detail: str = "Run cancelled",
    ) -> RuntimeRunSnapshot: ...

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
    "ExecutionScope",
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
    "RuntimeDeps",
    "RuntimeEventSink",
    "RuntimeFollowUpSink",
    "RuntimeStore",
    "Runtime",
    "RuntimeStoreSchemaError",
    "SessionConflictError",
    "SessionFormatError",
    "SessionLockTimeoutError",
    "SessionNotFoundError",
    "SessionPinMismatchError",
    "ScopedState",
    "StoredRunEvent",
    "agent_spec_digest",
    "ensure_session_pin",
    "new_session_id",
    "validate_namespace",
    "validate_session_id",
]

from .runtime import Runtime  # noqa: E402
