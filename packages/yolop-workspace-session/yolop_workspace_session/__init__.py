from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_core import from_json, to_json


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

    def _load(self, session_id: str) -> SessionSnapshot:
        content = self._path(session_id).read_bytes()
        raw_messages = [from_json(line) for line in content.splitlines()]
        messages = ModelMessagesTypeAdapter.validate_python(raw_messages)
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
        path = self._path(session_id)
        current = path.read_bytes()
        if _revision(current) != expected_revision:
            raise ValueError(f"Session {session_id!r} has changed")

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

    def _path(self, session_id: str) -> Path:
        return self._directory / f"{session_id}.jsonl"


def _revision(content: bytes) -> str:
    return sha256(content).hexdigest()


__all__ = ["SessionSnapshot", "WorkspaceSessionStore"]
