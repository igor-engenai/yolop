import json
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol, TypeVar
from uuid import UUID, uuid4

from pydantic_ai import AgentSpec, AgentStreamEvent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserContent, UserPromptPart
from pydantic_ai.usage import RunUsage

from .events import (
    AllEventsPersistencePolicy,
    RuntimeEventPersistencePolicy,
    SparseEventPersistencePolicy,
)


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


class RunBudgetExceededError(RuntimeError):
    """A root budget cannot admit more related work."""

    code = "run_budget_exceeded"


class RuntimeDeadlineExceededError(RuntimeError):
    """A root wall deadline has passed."""

    code = "runtime_deadline_exceeded"


class RuntimeStoreSchemaError(RuntimeError):
    """Stored runtime data uses an unsupported schema."""

    code = "runtime_store_schema_mismatch"


class CompactionUnsupportedError(RuntimeError):
    """The selected AgentSpec does not provide a manual compaction capability."""

    code = "compaction_unsupported"


class SessionLockTimeoutError(TimeoutError):
    """A session lock could not be acquired before its deadline."""

    code = "session_lock_timeout"


class StateSequenceConflictError(RuntimeError):
    """A state append used a stale sequence."""

    code = "state_sequence_conflict"


class StateFormatError(ValueError):
    """Stored plugin state is not valid bounded JSON state."""

    code = "state_format_error"


class StatePayloadLimitError(ValueError):
    """Plugin state exceeds the runtime payload limit."""

    code = "state_payload_limit"


class StateSchemaError(ValueError):
    """Plugin state does not use the requested schema version."""

    code = "state_schema_error"


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


class RuntimeEventSink(Protocol):
    """Receives native Pydantic AI stream events without translation."""

    async def emit(self, event: AgentStreamEvent) -> None: ...


class RuntimeContextSink(Protocol):
    """Receives the native RunContext for steering an active Run."""

    async def set_context(self, context: Any) -> None: ...


class RuntimeCompactor(Protocol):
    """Host/plugin compactor used by the generic manual Session operation."""

    async def compact(
        self,
        messages: Sequence[ModelMessage],
        *,
        focus: str | None,
        model: Any,
        deps: Any,
        scope: ExecutionScope,
    ) -> list[ModelMessage]: ...


class RuntimeFollowUpSink(Protocol):
    """Receives host follow-up prompts for the current execution."""

    async def enqueue(self, prompt: str) -> None: ...


HostDepsT = TypeVar("HostDepsT")
StateT = TypeVar("StateT")


@dataclass(frozen=True)
class RuntimeDeps[HostDepsT, StateT]:
    """Dependencies exposed to one runtime execution."""

    scope: ExecutionScope
    state: StateT
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
    """A namespaced session at one content revision and selected Run head."""

    id: str
    namespace: str
    pin: ExecutionPin
    messages: list[ModelMessage]
    revision: str
    head_run_id: str | None = None


class RunStatus(StrEnum):
    """Durable run lifecycle states."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RunRelation(StrEnum):
    """Relationship of a Run to its root execution."""

    ROOT = "root"
    CONTINUATION = "continuation"
    CHILD = "child"


@dataclass(frozen=True)
class RuntimeBudget:
    """Durable aggregate limits shared by a root Run and its descendants."""

    request_limit: int | None = None
    input_tokens_limit: int | None = None
    output_tokens_limit: int | None = None
    total_tokens_limit: int | None = None
    child_run_limit: int | None = None
    continuation_limit: int | None = None
    wall_deadline: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "request_limit",
            "input_tokens_limit",
            "output_tokens_limit",
            "total_tokens_limit",
            "child_run_limit",
            "continuation_limit",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"Runtime budget {name} must be non-negative")
        if self.wall_deadline is not None and self.wall_deadline.tzinfo is None:
            raise ValueError("Runtime budget wall_deadline must be timezone-aware")


@dataclass(frozen=True)
class RootBudgetSnapshot:
    """Durable root budget limits and consumed aggregate usage."""

    namespace: str
    root_run_id: str
    budget: RuntimeBudget
    requests_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    total_tokens_used: int = 0
    child_runs_used: int = 0
    continuations_used: int = 0
    active_runs: int = 0
    stopped: bool = False
    updated_at: datetime | None = None


class StateScope(StrEnum):
    """Durable plugin state lifetime."""

    SESSION = "session"
    RUN = "run"


MAX_STATE_OWNER_ID_LENGTH = 128
MAX_STATE_KIND_LENGTH = 128
MAX_STATE_PAYLOAD_BYTES = 64 * 1024


@dataclass(frozen=True)
class PluginStateEntry:
    """One append-only opaque plugin state entry."""

    namespace: str
    owner_id: str
    scope: StateScope
    scope_id: str
    state_kind: str
    schema_version: int
    sequence: int
    payload: Any
    created_at: datetime

    def __post_init__(self) -> None:
        validate_namespace(self.namespace)
        validate_session_id(self.scope_id)
        if not isinstance(self.scope, StateScope):
            try:
                object.__setattr__(self, "scope", StateScope(self.scope))
            except (TypeError, ValueError) as error:
                raise StateFormatError("State scope is unsupported") from error
        _validate_state_text(self.owner_id, "owner_id")
        _validate_state_text(self.state_kind, "state_kind")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise StateSchemaError("State schema version must be an integer")
        if self.schema_version < 1:
            raise StateSchemaError("State schema version must be positive")
        if not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("State sequence must be a positive integer")
        encode_state_payload(self.payload)


@dataclass(frozen=True)
class StoredRunEvent:
    """One ordered serialized event from a run."""

    sequence: int
    event: str
    data: str


@dataclass(frozen=True)
class RuntimeRunSnapshot:
    """An immutable durable Run history node."""

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
    parent_run_id: str | None = None
    root_run_id: str | None = None
    relation: RunRelation = RunRelation.ROOT
    initiator: str = "user"
    input_digest: str = ""
    full_messages: list[ModelMessage] = field(default_factory=list)
    active_messages: list[ModelMessage] = field(default_factory=list)
    events: list[StoredRunEvent] = field(default_factory=list)


@dataclass(frozen=True)
class RunTreeNode:
    """One immutable Run tree node for history navigation."""

    run: RuntimeRunSnapshot
    children: tuple["RunTreeNode", ...] = ()
    label: str | None = None
    selected: bool = False


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


def _validate_state_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"State {field_name} must not be empty")
    limit = MAX_STATE_OWNER_ID_LENGTH if field_name == "owner_id" else MAX_STATE_KIND_LENGTH
    if len(value) > limit:
        raise ValueError(f"State {field_name} exceeds {limit} characters")
    return value


def encode_state_payload(payload: Any) -> bytes:
    """Validate and encode an opaque JSON payload within the runtime limit."""
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise StateFormatError("State payload must be valid JSON") from error
    if len(encoded) > MAX_STATE_PAYLOAD_BYTES:
        raise StatePayloadLimitError(f"State payload exceeds {MAX_STATE_PAYLOAD_BYTES} bytes")
    return encoded


def decode_state_payload(encoded: bytes | str) -> Any:
    """Decode a stored JSON payload without silently repairing it."""
    if isinstance(encoded, str):
        encoded = encoded.encode()
    if not isinstance(encoded, bytes):
        raise StateFormatError("Stored state payload is not valid JSON")
    if len(encoded) > MAX_STATE_PAYLOAD_BYTES:
        raise StatePayloadLimitError(f"State payload exceeds {MAX_STATE_PAYLOAD_BYTES} bytes")
    try:
        payload = json.loads(encoded)
        encode_state_payload(payload)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError, ValueError) as error:
        if isinstance(error, StatePayloadLimitError):
            raise
        raise StateFormatError("Stored state payload is not valid JSON") from error
    return payload


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

    async def load_root_budget(
        self,
        namespace: str,
        root_run_id: str,
    ) -> RootBudgetSnapshot | None: ...

    async def read_state(
        self,
        namespace: str,
        *,
        owner_id: str,
        scope: StateScope,
        scope_id: str,
        state_kind: str,
        schema_version: int | None = None,
    ) -> list[PluginStateEntry]: ...

    async def append_state(
        self,
        namespace: str,
        *,
        owner_id: str,
        scope: StateScope,
        scope_id: str,
        state_kind: str,
        schema_version: int,
        expected_sequence: int,
        payload: Any,
    ) -> PluginStateEntry: ...

    async def checkout_session(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        *,
        expected_revision: str,
    ) -> RuntimeSessionSnapshot: ...

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
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
        relation: RunRelation = RunRelation.ROOT,
        root_budget: RuntimeBudget | None = None,
        initiator: str = "user",
        input_digest: str | None = None,
        full_messages: Sequence[ModelMessage] = (),
        active_messages: Sequence[ModelMessage] = (),
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
        expected_session_revision: str | None = None,
        full_messages: Sequence[ModelMessage] | None = None,
        active_messages: Sequence[ModelMessage] | None = None,
        output: Any | None = None,
        usage: RunUsage | None = None,
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
        messages: Sequence[ModelMessage] | None = None,
        full_messages: Sequence[ModelMessage] | None = None,
        active_messages: Sequence[ModelMessage] | None = None,
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


@dataclass(frozen=True)
class ScopedState:
    """Owner-bound access to one namespace and execution scope."""

    store: RuntimeStore
    namespace: str
    owner_id: str
    scope: StateScope
    scope_id: str

    async def read(
        self,
        state_kind: str,
        *,
        schema_version: int | None = 1,
    ) -> list[PluginStateEntry]:
        return await self.store.read_state(
            self.namespace,
            owner_id=self.owner_id,
            scope=self.scope,
            scope_id=self.scope_id,
            state_kind=state_kind,
            schema_version=schema_version,
        )

    async def append(
        self,
        state_kind: str,
        payload: Any,
        *,
        schema_version: int = 1,
        expected_sequence: int = 0,
    ) -> PluginStateEntry:
        return await self.store.append_state(
            self.namespace,
            owner_id=self.owner_id,
            scope=self.scope,
            scope_id=self.scope_id,
            state_kind=state_kind,
            schema_version=schema_version,
            expected_sequence=expected_sequence,
            payload=payload,
        )


@dataclass(frozen=True)
class ScopedStateContext:
    """Namespace and execution-bound state handle factory."""

    store: RuntimeStore
    namespace: str
    session_id: str
    run_id: str

    def bind(self, owner_id: str, *, scope: StateScope = StateScope.RUN) -> ScopedState:
        scope_id = self.run_id if scope is StateScope.RUN else self.session_id
        return ScopedState(
            store=self.store,
            namespace=self.namespace,
            owner_id=_validate_state_text(owner_id, "owner_id"),
            scope=scope,
            scope_id=scope_id,
        )

    def for_run(self, owner_id: str) -> ScopedState:
        return self.bind(owner_id, scope=StateScope.RUN)

    def for_session(self, owner_id: str) -> ScopedState:
        return self.bind(owner_id, scope=StateScope.SESSION)


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


def canonical_turn_messages(
    messages: Sequence[ModelMessage],
    prompt: str | Sequence[UserContent] | None,
) -> list[ModelMessage]:
    """Return only the current turn from a possibly rewritten active result.

    Compaction capabilities may rewrite prior active messages before Pydantic AI returns its
    result. The current user prompt is the durable boundary between that projection and the
    native messages produced by this Run.
    """
    if prompt is None:
        return list(messages)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, ModelRequest):
            continue
        if any(
            isinstance(part, UserPromptPart) and part.content == prompt for part in message.parts
        ):
            return list(messages[index:])
    return list(messages)


def input_digest(prompt: str) -> str:
    """Return a stable digest for persisted Run input."""
    return sha256(prompt.encode()).hexdigest()


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
    "RunBudgetExceededError",
    "RunCompletion",
    "RunNotFoundError",
    "RunRelation",
    "RunReservation",
    "RunStateError",
    "RunStatus",
    "RuntimeRunSnapshot",
    "RuntimeSessionSnapshot",
    "RunTreeNode",
    "RuntimeBudget",
    "RuntimeDeadlineExceededError",
    "RuntimeDeps",
    "RuntimeCompactor",
    "RuntimeContextSink",
    "RuntimeEventSink",
    "RuntimeEventPersistencePolicy",
    "AllEventsPersistencePolicy",
    "SparseEventPersistencePolicy",
    "RuntimeFollowUpSink",
    "RuntimeStore",
    "Runtime",
    "RuntimeStoreSchemaError",
    "CompactionUnsupportedError",
    "RootBudgetSnapshot",
    "ScopedStateContext",
    "SessionConflictError",
    "SessionFormatError",
    "SessionLockTimeoutError",
    "SessionNotFoundError",
    "SessionPinMismatchError",
    "PluginStateEntry",
    "ScopedState",
    "StateFormatError",
    "StatePayloadLimitError",
    "StateSchemaError",
    "StateScope",
    "StateSequenceConflictError",
    "StoredRunEvent",
    "decode_state_payload",
    "encode_state_payload",
    "agent_spec_digest",
    "canonical_turn_messages",
    "ensure_session_pin",
    "input_digest",
    "new_session_id",
    "validate_namespace",
    "validate_session_id",
]

from .runtime import Runtime  # noqa: E402
