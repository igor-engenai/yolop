from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from psycopg import AsyncConnection, Notify

_CHANNEL = "yolop_runtime_events"


class _NotificationStore(Protocol):
    _dsn: str
    _notifications_enabled: bool


async def _empty_notifications() -> AsyncIterator[Notify]:
    if False:
        yield Notify("", "", 0)


class PostgresNotificationOperations:
    """Optional PostgreSQL notification wake-up operations."""

    @asynccontextmanager
    async def event_notifications(
        self: _NotificationStore,
    ) -> AsyncGenerator[AsyncIterator[Notify], None]:
        if not self._notifications_enabled:
            yield _empty_notifications()
            return
        connection = await AsyncConnection.connect(self._dsn)
        try:
            await connection.execute(f"LISTEN {_CHANNEL}")
            await connection.commit()
            yield connection.notifies()
        finally:
            await connection.close()


__all__ = ["PostgresNotificationOperations"]
