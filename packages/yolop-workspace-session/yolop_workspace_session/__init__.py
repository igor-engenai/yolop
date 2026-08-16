from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from filelock import FileLock
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_core import from_json, to_json


class InvalidSessionIdError(ValueError):
    """A session ID is not a generated UUID4."""


class SessionConflictError(RuntimeError):
    """The stored session does not match the expected revision."""


class SessionFormatError(ValueError):
    """A stored session does not contain valid ModelMessage JSONL."""


class SessionNotFoundError(LookupError):
    """A generated session ID does not exist in this workspace."""


@dataclass(frozen=True)
class SessionSnapshot:
    """A session history at one content revision."""

    id: str
    messages: list[ModelMessage]
    revision: str


class WorkspaceSessionStore:
    """Store agent sessions below a host-provided workspace."""

    def __init__(self, workspace: str | Path) -> None:
        self._directory = Path(workspace).resolve() / ".yolop" / "sessions"

    async def create(self) -> SessionSnapshot:
        """Create and return an empty session with a generated ID."""
        return await asyncio.to_thread(self._create)

    async def list_sessions(self) -> list[str]:
        """List session IDs in stable order."""
        return await asyncio.to_thread(self._list_sessions)

    async def load(self, session_id: str) -> SessionSnapshot:
        """Load one session and its content revision."""
        return await asyncio.to_thread(self._load, session_id)

    async def delete(self, session_id: str, *, expected_revision: str) -> None:
        """Delete a session if its revision is current."""
        await asyncio.to_thread(self._delete, session_id, expected_revision)

    async def replace(
        self,
        session_id: str,
        *,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> SessionSnapshot:
        """Atomically replace a session's complete message history."""
        return await asyncio.to_thread(self._replace, session_id, expected_revision, messages)

    def _create(self) -> SessionSnapshot:
        self._directory.mkdir(parents=True, exist_ok=True)
        while True:
            session_id = str(uuid4())
            try:
                self._path(session_id).touch(exist_ok=False)
            except FileExistsError:
                continue
            return SessionSnapshot(
                id=session_id,
                messages=[],
                revision=sha256(b"").hexdigest(),
            )

    def _list_sessions(self) -> list[str]:
        if not self._directory.exists():
            return []
        return sorted(path.stem for path in self._directory.glob("*.jsonl"))

    def _delete(self, session_id: str, expected_revision: str) -> None:
        with FileLock(self._lock_path(session_id)):
            content = self._read(session_id)
            if _revision(content) != expected_revision:
                raise SessionConflictError(f"Session {session_id!r} has changed")
            self._path(session_id).unlink()

    def _load(self, session_id: str) -> SessionSnapshot:
        content = self._read(session_id)
        messages: list[ModelMessage] = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            try:
                messages.extend(ModelMessagesTypeAdapter.validate_python([from_json(line)]))
            except (ValueError, ValidationError) as error:
                raise SessionFormatError(
                    f"Session {session_id!r} has invalid JSONL at line {line_number}"
                ) from error
        return SessionSnapshot(
            id=session_id,
            messages=messages,
            revision=_revision(content),
        )

    def _replace(
        self,
        session_id: str,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> SessionSnapshot:
        with FileLock(self._lock_path(session_id)):
            path = self._path(session_id)
            current = self._read(session_id)
            if _revision(current) != expected_revision:
                raise SessionConflictError(f"Session {session_id!r} has changed")

            json_messages = ModelMessagesTypeAdapter.dump_python(list(messages), mode="json")
            content = b"".join(to_json(message) + b"\n" for message in json_messages)
            temporary = self._directory / f".{session_id}.{uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as file:
                    file.write(content)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

            return SessionSnapshot(
                id=session_id,
                messages=list(messages),
                revision=_revision(content),
            )

    def _lock_path(self, session_id: str) -> Path:
        return self._path(session_id).with_suffix(".lock")

    def _read(self, session_id: str) -> bytes:
        try:
            return self._path(session_id).read_bytes()
        except FileNotFoundError as error:
            raise SessionNotFoundError(f"Session {session_id!r} does not exist") from error

    def _path(self, session_id: str) -> Path:
        try:
            parsed = UUID(session_id)
        except (AttributeError, TypeError, ValueError) as error:
            raise InvalidSessionIdError(
                f"Session ID {session_id!r} is not a generated UUID4"
            ) from error
        if parsed.version != 4 or str(parsed) != session_id:
            raise InvalidSessionIdError(f"Session ID {session_id!r} is not a generated UUID4")
        return self._directory / f"{session_id}.jsonl"


def _revision(content: bytes) -> str:
    return sha256(content).hexdigest()


__all__ = [
    "InvalidSessionIdError",
    "SessionConflictError",
    "SessionFormatError",
    "SessionNotFoundError",
    "SessionSnapshot",
    "WorkspaceSessionStore",
]
