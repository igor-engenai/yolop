from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from yolop_postgres_runtime import PostgresRuntimeStore
from yolop_runtime import RuntimeStore

_CONTRACT_DIR = Path(__file__).resolve().parents[2] / "yolop-runtime" / "tests" / "contract"
sys.path.insert(0, str(_CONTRACT_DIR))

from runtime_store_contract import (  # noqa: E402  # ty: ignore[unresolved-import]
    RuntimeStoreContract,
)


class TestPostgresRuntimeStoreContract(RuntimeStoreContract):
    _dsn: str
    _stores: list[PostgresRuntimeStore]

    @pytest.fixture(autouse=True)
    async def configure_store(self, postgres_dsn: str) -> AsyncIterator[None]:
        self._dsn = postgres_dsn
        self._stores = []
        yield
        for store in self._stores:
            await store.close()

    async def make_store(self, database: Path) -> RuntimeStore:
        del database
        store = await PostgresRuntimeStore(self._dsn).open()
        self._stores.append(store)
        return store
