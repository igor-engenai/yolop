from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID, uuid4

from pydantic_ai.messages import ModelMessage


class InvalidSessionIdError(ValueError):
    """A session ID is not a generated UUID4."""


class SessionConflictError(RuntimeError):
    """The stored session does not match the expected revision."""


class SessionFormatError(ValueError):
    """Stored session messages are not valid Pydantic AI messages."""


class SessionNotFoundError(LookupError):
    """A session ID does not exist in the selected store."""


@dataclass(frozen=True)
class SessionSnapshot:
    """A session history at one content revision."""

    id: str
    messages: list[ModelMessage]
    revision: str


class SessionStore(Protocol):
    """Persistence operations required by a YoloP session host."""

    async def create(self) -> SessionSnapshot:
        """Create and return an empty session with a generated ID."""
        ...

    async def list_sessions(self) -> list[str]:
        """List session IDs in stable order."""
        ...

    async def load(self, session_id: str) -> SessionSnapshot:
        """Load one session and its content revision."""
        ...

    async def delete(self, session_id: str, *, expected_revision: str) -> None:
        """Delete a session if its revision is current."""
        ...

    async def replace(
        self,
        session_id: str,
        *,
        expected_revision: str,
        messages: Sequence[ModelMessage],
    ) -> SessionSnapshot:
        """Atomically replace a session's complete message history."""
        ...


def new_session_id() -> str:
    """Return a canonical generated UUID4 session ID."""
    return str(uuid4())


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
    "InvalidSessionIdError",
    "SessionConflictError",
    "SessionFormatError",
    "SessionNotFoundError",
    "SessionSnapshot",
    "SessionStore",
    "new_session_id",
    "validate_session_id",
]
