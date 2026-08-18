import sys
from pathlib import Path

from yolop_runtime import RuntimeStore
from yolop_sqlite_session import SQLiteRuntimeStore

_CONTRACT_DIR = Path(__file__).resolve().parents[2] / "yolop-runtime" / "tests" / "contract"
sys.path.insert(0, str(_CONTRACT_DIR))

from runtime_store_contract import (  # noqa: E402  # ty: ignore[unresolved-import]
    RuntimeStoreContract,
)


class TestSQLiteRuntimeStoreContract(RuntimeStoreContract):
    async def make_store(self, database: Path) -> RuntimeStore:
        return SQLiteRuntimeStore(database)
