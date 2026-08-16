from pathlib import Path

from yolop_sqlite_session import SQLiteRuntimeStore


class WorkspaceRuntimeStore(SQLiteRuntimeStore):
    """Store namespaced runtime state below a host-provided workspace."""

    def __init__(self, workspace: str | Path) -> None:
        root = Path(workspace).expanduser().resolve()
        super().__init__(root / ".yolop" / "runtime.db")


__all__ = ["WorkspaceRuntimeStore"]
