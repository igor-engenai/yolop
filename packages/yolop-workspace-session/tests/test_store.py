from uuid import uuid4

from yolop_session import ExecutionPin
from yolop_workspace_session import WorkspaceRuntimeStore


async def test_workspace_runtime_store_uses_one_sqlite_database(tmp_path) -> None:
    old_directory = tmp_path / ".yolop" / "sessions"
    old_directory.mkdir(parents=True)
    (old_directory / f"{uuid4()}.jsonl").write_text("")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    store = WorkspaceRuntimeStore(tmp_path)

    session = await store.create_session("local", pin=pin)
    reopened = WorkspaceRuntimeStore(tmp_path)

    assert (tmp_path / ".yolop" / "runtime.db").is_file()
    assert await reopened.list_sessions("local") == [session.id]
    assert await reopened.load_session("local", session.id) == session
