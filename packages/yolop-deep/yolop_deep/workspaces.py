"""Host-owned isolated workspaces for experimental fork candidates."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from yolop_runtime import Runtime, ScopedStateContext, validate_session_id

_WORKSPACE_OWNER = "yolop.deep.fork_workspaces"
_WORKSPACE_STATE_KIND = "workspaces"
_WORKSPACE_SCHEMA_VERSION = 1


class WorkspaceIsolationError(ValueError):
    """A candidate workspace violates host path or lifecycle policy."""

    code = "fork_workspace_isolation"


class CandidateWorkspaceLimitError(ValueError):
    """The host candidate workspace limit has been reached."""

    code = "fork_workspace_limit"


class _WorkspaceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_key: str = Field(min_length=1, max_length=255)
    namespace: str
    source_session_id: str
    candidate_session_id: str
    source_workspace: str
    path: str


@dataclass(frozen=True)
class CandidateWorkspaceHandle:
    """Safe host handle for one isolated candidate workspace."""

    namespace: str
    candidate_key: str
    source_session_id: str
    candidate_session_id: str
    path: Path


class _WorkspaceState:
    def __init__(self, runtime: Runtime[Any], namespace: str, session_id: str, run_id: str) -> None:
        self._state = ScopedStateContext(
            store=runtime.store,
            namespace=namespace,
            session_id=session_id,
            run_id=run_id,
        ).for_session(_WORKSPACE_OWNER)

    async def records(self) -> dict[str, _WorkspaceRecord]:
        entries = await self._state.read(
            _WORKSPACE_STATE_KIND, schema_version=_WORKSPACE_SCHEMA_VERSION
        )
        if not entries:
            return {}
        payload = entries[-1].payload
        if not isinstance(payload, dict) or not isinstance(payload.get("workspaces"), dict):
            raise WorkspaceIsolationError("Stored candidate workspace state is invalid")
        try:
            return {
                key: _WorkspaceRecord.model_validate(value)
                for key, value in payload["workspaces"].items()
            }
        except (TypeError, ValueError) as error:
            raise WorkspaceIsolationError("Stored candidate workspace state is invalid") from error

    async def write(self, records: dict[str, _WorkspaceRecord]) -> None:
        entries = await self._state.read(
            _WORKSPACE_STATE_KIND, schema_version=_WORKSPACE_SCHEMA_VERSION
        )
        expected_sequence = entries[-1].sequence if entries else 0
        await self._state.append(
            _WORKSPACE_STATE_KIND,
            {"workspaces": {key: value.model_dump(mode="json") for key, value in records.items()}},
            schema_version=_WORKSPACE_SCHEMA_VERSION,
            expected_sequence=expected_sequence,
        )


class CandidateWorkspaceService:
    """Allocate, reconnect, and remove isolated candidate workspace copies."""

    def __init__(
        self,
        runtime: Runtime[Any],
        *,
        root: str | Path,
        max_candidates: int = 4,
    ) -> None:
        if isinstance(max_candidates, bool) or max_candidates < 1:
            raise CandidateWorkspaceLimitError("Candidate workspace limit must be positive")
        self.runtime = runtime
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise WorkspaceIsolationError("Candidate workspace root must be a real directory")
        self.max_candidates = max_candidates

    async def allocate(
        self,
        namespace: str,
        source_session_id: str,
        source_run_id: str,
        *,
        candidate_session_id: str,
        candidate_key: str,
        source_workspace: str | Path,
    ) -> CandidateWorkspaceHandle:
        validate_session_id(source_session_id)
        validate_session_id(candidate_session_id)
        if not candidate_key.strip() or len(candidate_key) > 255:
            raise WorkspaceIsolationError("Candidate workspace key is empty or too long")
        source = self._authorized_source(source_workspace)
        template = _WorkspaceRecord(
            candidate_key=candidate_key,
            namespace=namespace,
            source_session_id=source_session_id,
            candidate_session_id=candidate_session_id,
            source_workspace=str(source),
            path=str(self._candidate_path(namespace, candidate_session_id)),
        )
        state = _WorkspaceState(self.runtime, namespace, source_session_id, source_run_id)
        async with self.runtime.store.lock_session(namespace, source_session_id, timeout=30):
            records = await state.records()
            existing = records.get(candidate_key)
            if existing is not None:
                if (
                    existing.candidate_session_id != candidate_session_id
                    or existing.source_workspace != str(source)
                ):
                    raise WorkspaceIsolationError("Candidate workspace key has different input")
                return _handle(existing)
            if len(records) >= self.max_candidates:
                raise CandidateWorkspaceLimitError("Candidate workspace limit has been reached")
            destination = Path(template.path)
            if destination.exists():
                raise WorkspaceIsolationError("Candidate workspace destination already exists")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_without_symlinks(source, destination)
            await state.write({**records, candidate_key: template})
            return _handle(template)

    async def inspect(self, handle: CandidateWorkspaceHandle) -> CandidateWorkspaceHandle:
        record = await self._record(handle)
        path = Path(record.path)
        self._ensure_candidate_path(path, record.candidate_session_id)
        return _handle(record)

    async def list_workspaces(
        self,
        namespace: str,
        source_session_id: str,
        source_run_id: str,
    ) -> list[CandidateWorkspaceHandle]:
        records = await _WorkspaceState(
            self.runtime, namespace, source_session_id, source_run_id
        ).records()
        return [_handle(record) for record in records.values()]

    async def cleanup(self, handle: CandidateWorkspaceHandle) -> None:
        record = await self._record(handle)
        path = Path(record.path)
        self._ensure_candidate_path(path, record.candidate_session_id)
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise WorkspaceIsolationError("Candidate workspace is not a real directory")
            shutil.rmtree(path)
        state = _WorkspaceState(
            self.runtime,
            record.namespace,
            record.source_session_id,
            record.source_session_id,
        )
        async with self.runtime.store.lock_session(
            record.namespace, record.source_session_id, timeout=30
        ):
            records = await state.records()
            records.pop(record.candidate_key, None)
            await state.write(records)

    async def _record(self, handle: CandidateWorkspaceHandle) -> _WorkspaceRecord:
        template = _WorkspaceRecord(
            candidate_key=handle.candidate_key,
            namespace=handle.namespace,
            source_session_id=handle.source_session_id,
            candidate_session_id=handle.candidate_session_id,
            source_workspace=str(self.root),
            path=str(handle.path),
        )
        record = (
            await _WorkspaceState(
                self.runtime,
                handle.namespace,
                handle.source_session_id,
                handle.source_session_id,
            ).records()
        ).get(handle.candidate_key)
        if record is None or record.candidate_session_id != template.candidate_session_id:
            raise WorkspaceIsolationError("Candidate workspace handle is not available")
        return record

    def _authorized_source(self, source_workspace: str | Path) -> Path:
        source_path = Path(source_workspace).expanduser()
        if source_path.is_symlink():
            raise WorkspaceIsolationError("Source workspace must not be a symlink")
        try:
            source = source_path.resolve(strict=True)
        except OSError as error:
            raise WorkspaceIsolationError("Source workspace does not exist") from error
        if not source.is_dir() or not _within(self.root, source):
            raise WorkspaceIsolationError("Source workspace is outside the authorized root")
        return source

    def _candidate_path(self, namespace: str, candidate_session_id: str) -> Path:
        namespace_key = hashlib.sha256(namespace.encode()).hexdigest()[:32]
        path = self.root / ".yolop" / "candidates" / namespace_key / candidate_session_id
        self._ensure_candidate_path(path, candidate_session_id)
        return path

    def _ensure_candidate_path(self, path: Path, candidate_session_id: str) -> None:
        root = (self.root / ".yolop" / "candidates").resolve()
        resolved_parent = path.parent.resolve()
        if path.name != candidate_session_id or not _within(root, resolved_parent):
            raise WorkspaceIsolationError("Candidate path is outside the authorized root")


def _copy_without_symlinks(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.name == ".yolop":
            continue
        if item.is_symlink():
            raise WorkspaceIsolationError("Candidate copy rejects symlink entries")
        target = destination / item.name
        if item.is_dir():
            _copy_without_symlinks(item, target)
        else:
            shutil.copy2(item, target)


def _within(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _handle(record: _WorkspaceRecord) -> CandidateWorkspaceHandle:
    return CandidateWorkspaceHandle(
        namespace=record.namespace,
        candidate_key=record.candidate_key,
        source_session_id=record.source_session_id,
        candidate_session_id=record.candidate_session_id,
        path=Path(record.path),
    )


__all__ = [
    "CandidateWorkspaceHandle",
    "CandidateWorkspaceLimitError",
    "CandidateWorkspaceService",
    "WorkspaceIsolationError",
]
