import asyncio
import json
from uuid import UUID, uuid4

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pytest import raises
from yolop_workspace_session import (
    InvalidSessionIdError,
    SessionConflictError,
    SessionFormatError,
    SessionNotFoundError,
    SessionSnapshot,
    WorkspaceSessionStore,
)


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


async def test_replace_rejects_a_stale_revision_without_changing_history(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)
    session = await store.create()
    saved = await store.replace(
        session.id,
        expected_revision=session.revision,
        messages=[ModelRequest(parts=[UserPromptPart("Saved")])],
    )

    with raises(SessionConflictError, match=session.id):
        await store.replace(
            session.id,
            expected_revision=session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("Lost")])],
        )

    assert await store.load(session.id) == saved
    assert not list((tmp_path / ".yolop" / "sessions").glob("*.tmp"))


async def test_load_reports_the_line_of_malformed_jsonl(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)
    session = await store.create()
    await store.replace(
        session.id,
        expected_revision=session.revision,
        messages=[ModelRequest(parts=[UserPromptPart("Valid")])],
    )
    path = tmp_path / ".yolop" / "sessions" / f"{session.id}.jsonl"
    with path.open("a") as file:
        file.write("not-json\n")

    with raises(SessionFormatError, match=rf"{session.id!s}.*line 2"):
        await store.load(session.id)


async def test_load_rejects_an_invalid_session_id(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)

    with raises(InvalidSessionIdError, match="not a generated UUID4"):
        await store.load("../../outside")


async def test_load_reports_an_unknown_generated_session_id(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)
    unknown_id = str(uuid4())

    with raises(SessionNotFoundError, match=unknown_id):
        await store.load(unknown_id)


async def test_delete_requires_the_current_revision(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)
    session = await store.create()
    updated = await store.replace(
        session.id,
        expected_revision=session.revision,
        messages=[ModelRequest(parts=[UserPromptPart("Saved")])],
    )

    with raises(SessionConflictError, match=session.id):
        await store.delete(session.id, expected_revision=session.revision)

    await store.delete(session.id, expected_revision=updated.revision)
    assert await store.list_sessions() == []


async def test_only_one_concurrent_replace_can_use_a_revision(tmp_path) -> None:
    store = WorkspaceSessionStore(tmp_path)
    session = await store.create()

    results = await asyncio.gather(
        store.replace(
            session.id,
            expected_revision=session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("First")])] * 1_000,
        ),
        store.replace(
            session.id,
            expected_revision=session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("Second")])] * 1_000,
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, SessionSnapshot) for result in results) == 1
    assert sum(isinstance(result, SessionConflictError) for result in results) == 1
