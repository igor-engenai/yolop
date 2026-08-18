from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from .claims import PostgresClaimOperations
from .migrations import CURRENT_SCHEMA_VERSION, migrate
from .notifications import PostgresNotificationOperations
from .operations import PostgresSessionOperations, PostgresStateOperations
from .runs import PostgresRunOperations
from .terminal import PostgresTerminalOperations


@dataclass(frozen=True)
class PostgresPoolConfig:
    """Connection-pool settings owned by the embedding host."""

    min_size: int = 1
    max_size: int = 10
    timeout: float = 30.0
    notifications: bool = True

    def __post_init__(self) -> None:
        if self.min_size < 0:
            raise ValueError("PostgreSQL pool min_size cannot be negative")
        if self.max_size < 1 or self.max_size < self.min_size:
            raise ValueError("PostgreSQL pool max_size must be at least min_size")
        if self.timeout <= 0:
            raise ValueError("PostgreSQL pool timeout must be positive")


class PostgresRuntimeStore(
    PostgresSessionOperations,
    PostgresStateOperations,
    PostgresRunOperations,
    PostgresClaimOperations,
    PostgresTerminalOperations,
    PostgresNotificationOperations,
):
    """Production PostgreSQL connection boundary for YoloP RuntimeStore."""

    def __init__(
        self,
        dsn: str,
        *,
        pool: PostgresPoolConfig | None = None,
    ) -> None:
        if not dsn.strip():
            raise ValueError("PostgreSQL DSN must not be empty")
        self._dsn = dsn
        self._pool_config = pool or PostgresPoolConfig()
        self._notifications_enabled = self._pool_config.notifications
        self._pool: AsyncConnectionPool[AsyncConnection[Any]] = AsyncConnectionPool(
            conninfo=dsn,
            min_size=self._pool_config.min_size,
            max_size=self._pool_config.max_size,
            timeout=self._pool_config.timeout,
            open=False,
        )

    @property
    def pool(self) -> AsyncConnectionPool[AsyncConnection[Any]]:
        """Return the explicitly opened pool used by RuntimeStore operations."""
        return self._pool

    async def open(self) -> Self:
        """Open the connection pool without running database migrations."""
        await self._pool.open()
        return self

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "PostgresPoolConfig",
    "PostgresRuntimeStore",
    "migrate",
]
