from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from yolop_runtime import (
    RuntimeStore,
    StateScope,
    StateSequenceConflictError,
    validate_session_id,
)

_STATE_OWNER = "yolop.memory"
_STATE_KIND = "items"
_STATE_SCHEMA_VERSION = 1


class MemoryError(RuntimeError):
    """Base error for scoped persistent memory."""

    code = "memory_error"


class MemoryValidationError(ValueError, MemoryError):
    """Memory input is invalid or exceeds host bounds."""

    code = "memory_validation_error"


class MemoryNotFoundError(LookupError, MemoryError):
    """The requested active memory item does not exist."""

    code = "memory_not_found"


class MemoryRevisionConflictError(MemoryError):
    """A memory update used a stale revision."""

    code = "memory_revision_conflict"


class MemoryDestructiveOperationDeniedError(PermissionError, MemoryError):
    """The host did not authorize memory retirement."""

    code = "memory_destructive_operation_denied"


class MemoryScopeKind(StrEnum):
    USER = "user"
    WORKSPACE = "workspace"
    AGENT = "agent"
    SESSION = "session"


@dataclass(frozen=True)
class MemoryScope:
    """An explicit host-selected memory scope."""

    kind: MemoryScopeKind
    key: str

    def __post_init__(self) -> None:
        try:
            kind = MemoryScopeKind(self.kind)
        except (TypeError, ValueError) as error:
            raise MemoryValidationError("Memory scope kind is unsupported") from error
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.key, str) or not self.key.strip() or len(self.key) > 255:
            raise MemoryValidationError("Memory scope key must be bounded and non-empty")
        if self.kind is MemoryScopeKind.SESSION:
            try:
                validate_session_id(self.key)
            except Exception as error:
                raise MemoryValidationError("Session memory scope requires a UUID4 key") from error

    @classmethod
    def user(cls, key: str) -> MemoryScope:
        return cls(MemoryScopeKind.USER, key)

    @classmethod
    def workspace(cls, key: str) -> MemoryScope:
        return cls(MemoryScopeKind.WORKSPACE, key)

    @classmethod
    def agent(cls, key: str) -> MemoryScope:
        return cls(MemoryScopeKind.AGENT, key)

    @classmethod
    def session(cls, session_id: str) -> MemoryScope:
        return cls(MemoryScopeKind.SESSION, session_id)

    def state_scope_id(self, namespace: str) -> str:
        """Return the opaque UUID4 anchor used by RuntimeStore plugin state."""
        if self.kind is MemoryScopeKind.SESSION:
            return self.key
        digest = bytearray(
            sha256(f"yolop-memory-v1\0{namespace}\0{self.kind}\0{self.key}".encode()).digest()[:16]
        )
        digest[6] = (digest[6] & 0x0F) | 0x40
        digest[8] = (digest[8] & 0x3F) | 0x80
        return str(UUID(bytes=bytes(digest)))


@dataclass(frozen=True)
class MemoryLimits:
    max_content_bytes: int = 8 * 1024
    max_title_bytes: int = 256
    max_tags: int = 16
    max_tag_bytes: int = 64
    max_provenance_bytes: int = 512
    max_results: int = 20
    max_result_bytes: int = 32 * 1024
    allow_retire: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_content_bytes",
            "max_title_bytes",
            "max_tags",
            "max_tag_bytes",
            "max_provenance_bytes",
            "max_results",
            "max_result_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MemoryValidationError(f"Memory limit {name} must be positive")


@dataclass(frozen=True)
class MemoryRecord:
    """One auditable memory revision."""

    memory_id: str
    scope: MemoryScopeKind
    content: str
    title: str
    tags: tuple[str, ...]
    created_by_run_id: str
    created_at: datetime
    updated_at: datetime
    provenance: str
    revision: int = 1
    status: Literal["active", "superseded", "retired"] = "active"
    superseded_by: int | None = None

    @property
    def active(self) -> bool:
        return self.status == "active"

    @property
    def revision_id(self) -> str:
        return f"{self.memory_id}:{self.revision}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "scope": self.scope.value,
            "content": self.content,
            "title": self.title,
            "tags": list(self.tags),
            "created_by_run_id": self.created_by_run_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "provenance": self.provenance,
            "revision": self.revision,
            "status": self.status,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MemoryRecord:
        try:
            created_at = datetime.fromisoformat(str(payload["created_at"]))
            updated_at = datetime.fromisoformat(str(payload["updated_at"]))
            if created_at.tzinfo is None or updated_at.tzinfo is None:
                raise ValueError("timestamps must be timezone-aware")
            return cls(
                memory_id=str(payload["memory_id"]),
                scope=MemoryScopeKind(payload["scope"]),
                content=str(payload["content"]),
                title=str(payload.get("title", "")),
                tags=tuple(str(tag) for tag in payload.get("tags", ())),
                created_by_run_id=str(payload["created_by_run_id"]),
                created_at=created_at,
                updated_at=updated_at,
                provenance=str(payload["provenance"]),
                revision=int(payload["revision"]),
                status=payload.get("status", "active"),
                superseded_by=payload.get("superseded_by"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MemoryValidationError("Stored memory payload is invalid") from error


class MemoryStore(Protocol):
    async def create(
        self,
        content: str,
        *,
        created_by_run_id: str,
        provenance: str,
        title: str = "",
        tags: Sequence[str] = (),
    ) -> MemoryRecord: ...

    async def get(self, memory_id: str) -> MemoryRecord | None: ...

    async def history(self, memory_id: str) -> tuple[MemoryRecord, ...]: ...

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        include_inactive: bool = False,
    ) -> tuple[MemoryRecord, ...]: ...

    async def replace(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        content: str,
        updated_by_run_id: str,
        provenance: str,
        title: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> MemoryRecord: ...

    async def retire(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        retired_by_run_id: str,
        provenance: str,
    ) -> MemoryRecord: ...


class RuntimeMemoryStore:
    """Durable local MemoryStore backed by RuntimeStore plugin state."""

    def __init__(
        self,
        runtime_store: RuntimeStore,
        *,
        namespace: str,
        scope: MemoryScope,
        limits: MemoryLimits | None = None,
    ) -> None:
        self.runtime_store = runtime_store
        self.namespace = namespace
        self.scope = scope
        self.limits = limits or MemoryLimits()
        self._scope_id = scope.state_scope_id(namespace)

    async def create(
        self,
        content: str,
        *,
        created_by_run_id: str,
        provenance: str,
        title: str = "",
        tags: Sequence[str] = (),
    ) -> MemoryRecord:
        now = datetime.now(UTC)
        record = MemoryRecord(
            memory_id=str(uuid4()),
            scope=self.scope.kind,
            content=_validate_content(content, self.limits),
            title=_validate_title(title, self.limits),
            tags=_validate_tags(tags, self.limits),
            created_by_run_id=_validate_run_id(created_by_run_id),
            created_at=now,
            updated_at=now,
            provenance=_validate_provenance(provenance, self.limits),
        )
        await self._append({"op": "create", "record": record.to_payload()})
        return record

    async def get(self, memory_id: str) -> MemoryRecord | None:
        records = await self._records(memory_id)
        if not records or not records[-1].active:
            return None
        return records[-1]

    async def history(self, memory_id: str) -> tuple[MemoryRecord, ...]:
        return await self._records(memory_id)

    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        include_inactive: bool = False,
    ) -> tuple[MemoryRecord, ...]:
        terms = tuple(term.casefold() for term in query.split() if term.strip())
        if not terms:
            raise MemoryValidationError("Memory search query must not be empty")
        records = await self._all_records()
        latest: dict[str, MemoryRecord] = {}
        for record in records:
            latest[record.memory_id] = record
        candidates = [record for record in latest.values() if include_inactive or record.active]
        candidates = [record for record in candidates if _matches(record, terms)]
        candidates.sort(key=lambda record: (record.updated_at, record.memory_id), reverse=True)
        effective_limit = (
            self.limits.max_results if limit is None else min(limit, self.limits.max_results)
        )
        if effective_limit < 1:
            raise MemoryValidationError("Memory search limit must be positive")
        results: list[MemoryRecord] = []
        total_bytes = 0
        for record in candidates[:effective_limit]:
            encoded_size = len(json.dumps(record.to_payload(), ensure_ascii=False).encode())
            if total_bytes + encoded_size > self.limits.max_result_bytes:
                break
            results.append(record)
            total_bytes += encoded_size
        return tuple(results)

    async def replace(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        content: str,
        updated_by_run_id: str,
        provenance: str,
        title: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> MemoryRecord:
        current = await self.get(memory_id)
        if current is None:
            raise MemoryNotFoundError(f"Memory item {memory_id!r} does not exist")
        if current.revision != expected_revision:
            raise MemoryRevisionConflictError("Memory revision is stale")
        now = datetime.now(UTC)
        replacement = MemoryRecord(
            memory_id=current.memory_id,
            scope=current.scope,
            content=_validate_content(content, self.limits),
            title=current.title if title is None else _validate_title(title, self.limits),
            tags=current.tags if tags is None else _validate_tags(tags, self.limits),
            created_by_run_id=_validate_run_id(updated_by_run_id),
            created_at=current.created_at,
            updated_at=now,
            provenance=_validate_provenance(provenance, self.limits),
            revision=current.revision + 1,
        )
        current_payload = {
            **current.to_payload(),
            "status": "superseded",
            "superseded_by": replacement.revision,
        }
        await self._append(
            {"op": "replace", "previous": current_payload, "record": replacement.to_payload()}
        )
        return replacement

    async def retire(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        retired_by_run_id: str,
        provenance: str,
    ) -> MemoryRecord:
        if not self.limits.allow_retire:
            raise MemoryDestructiveOperationDeniedError("Memory retirement is not host-authorized")
        current = await self.get(memory_id)
        if current is None:
            raise MemoryNotFoundError(f"Memory item {memory_id!r} does not exist")
        if current.revision != expected_revision:
            raise MemoryRevisionConflictError("Memory revision is stale")
        retired = MemoryRecord(
            memory_id=current.memory_id,
            scope=current.scope,
            content=current.content,
            title=current.title,
            tags=current.tags,
            created_by_run_id=_validate_run_id(retired_by_run_id),
            created_at=current.created_at,
            updated_at=datetime.now(UTC),
            provenance=_validate_provenance(provenance, self.limits),
            revision=current.revision + 1,
            status="retired",
        )
        await self._append(
            {"op": "retire", "previous": current.to_payload(), "record": retired.to_payload()}
        )
        return retired

    async def _records(self, memory_id: str) -> tuple[MemoryRecord, ...]:
        records = [record for record in await self._all_records() if record.memory_id == memory_id]
        records.sort(key=lambda record: record.revision)
        return tuple(records)

    async def _all_records(self) -> tuple[MemoryRecord, ...]:
        entries = await self.runtime_store.read_state(
            self.namespace,
            owner_id=_STATE_OWNER,
            scope=StateScope.SESSION,
            scope_id=self._scope_id,
            state_kind=_STATE_KIND,
            schema_version=_STATE_SCHEMA_VERSION,
        )
        records: dict[str, MemoryRecord] = {}
        for entry in entries:
            payload = entry.payload
            if not isinstance(payload, Mapping):
                raise MemoryValidationError("Stored memory event is invalid")
            op = payload.get("op")
            if op == "create":
                record = MemoryRecord.from_payload(_mapping(payload.get("record")))
                records[record.revision_id] = record
            elif op in {"replace", "retire"}:
                previous = MemoryRecord.from_payload(_mapping(payload.get("previous")))
                record = MemoryRecord.from_payload(_mapping(payload.get("record")))
                records[previous.revision_id] = previous
                records[record.revision_id] = record
            else:
                raise MemoryValidationError("Stored memory event operation is invalid")
        return tuple(
            sorted(records.values(), key=lambda record: (record.updated_at, record.revision_id))
        )

    async def _append(self, payload: Mapping[str, Any]) -> None:
        entries = await self.runtime_store.read_state(
            self.namespace,
            owner_id=_STATE_OWNER,
            scope=StateScope.SESSION,
            scope_id=self._scope_id,
            state_kind=_STATE_KIND,
            schema_version=_STATE_SCHEMA_VERSION,
        )
        expected_sequence = entries[-1].sequence if entries else 0
        try:
            await self.runtime_store.append_state(
                self.namespace,
                owner_id=_STATE_OWNER,
                scope=StateScope.SESSION,
                scope_id=self._scope_id,
                state_kind=_STATE_KIND,
                schema_version=_STATE_SCHEMA_VERSION,
                expected_sequence=expected_sequence,
                payload=dict(payload),
            )
        except StateSequenceConflictError:
            raise


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MemoryValidationError("Stored memory record is invalid")
    return value


def _validate_content(value: str, limits: MemoryLimits) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError("Memory content must not be empty")
    if "\x00" in value or len(value.encode()) > limits.max_content_bytes:
        raise MemoryValidationError("Memory content exceeds the limit")
    return value


def _validate_title(value: str, limits: MemoryLimits) -> str:
    if not isinstance(value, str) or len(value.encode()) > limits.max_title_bytes:
        raise MemoryValidationError("Memory title exceeds the limit")
    return value


def _validate_tags(values: Sequence[str], limits: MemoryLimits) -> tuple[str, ...]:
    tags = tuple(values)
    if len(tags) > limits.max_tags or any(
        not isinstance(tag, str) or not tag.strip() or len(tag.encode()) > limits.max_tag_bytes
        for tag in tags
    ):
        raise MemoryValidationError("Memory tags exceed the limit")
    if len(set(tags)) != len(tags):
        raise MemoryValidationError("Memory tags must be unique")
    return tags


def _validate_provenance(value: str, limits: MemoryLimits) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode()) > limits.max_provenance_bytes
    ):
        raise MemoryValidationError("Memory provenance exceeds the limit")
    return value


def _validate_run_id(value: str) -> str:
    try:
        return validate_session_id(value)
    except Exception as error:
        raise MemoryValidationError("Memory writer Run ID must be a UUID4") from error


def _matches(record: MemoryRecord, terms: Sequence[str]) -> bool:
    haystack = " ".join((record.title, record.content, *record.tags)).casefold()
    return all(term in haystack for term in terms)
