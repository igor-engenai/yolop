import json
from uuid import UUID

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from yolop_workspace_session import WorkspaceSessionStore


async def test_create_returns_an_empty_generated_session(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)

    session = await store.create()

    assert UUID(session.id).version == 4
    assert session.messages == []
    assert session.revision


async def test_list_returns_multiple_generated_sessions(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)
    first = await store.create()
    second = await store.create()

    session_ids = await store.list_sessions()

    assert session_ids == sorted([first.id, second.id])


async def test_replace_and_load_preserve_native_messages(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)
    session = await store.create()
    messages = [
        ModelRequest(parts=[UserPromptPart("Question")]),
        ModelResponse(parts=[TextPart("Answer")]),
    ]

    updated = await store.replace(
        session.id,
        expected_revision=session.revision,
        messages=messages,
    )
    loaded = await store.load(session.id)

    assert updated.revision != session.revision
    assert loaded == updated
    assert loaded.messages == messages


async def test_session_file_contains_one_message_object_per_line(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)
    session = await store.create()
    messages = [
        ModelRequest(parts=[UserPromptPart("Question")]),
        ModelResponse(parts=[TextPart("Answer")]),
    ]

    await store.replace(
        session.id,
        expected_revision=session.revision,
        messages=messages,
    )

    lines = (tmp_path / ".yolop" / "sessions" / f"{session.id}.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["request", "response"]
