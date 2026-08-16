import asyncio
from uuid import UUID

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RunUsage
from pytest import raises
from yolop_session import (
    ExecutionPin,
    IdempotencyConflictError,
    RunAdmissionError,
    RunStateError,
    RunStatus,
    SessionConflictError,
)
from yolop_sqlite_session import SQLiteRuntimeStore


async def test_run_reservation_is_idempotent(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)

    first = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="Hello",
    )
    second = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="Hello",
    )

    assert first.created is True
    assert second.created is False
    assert second.run == first.run
    assert UUID(first.run.id).version == 4
    assert first.run.status is RunStatus.ACCEPTED


async def test_idempotency_key_rejects_different_input(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)
    await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="First",
    )

    with raises(IdempotencyConflictError) as raised:
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request-1",
            prompt="Different",
        )

    assert raised.value.code == "idempotency_conflict"


async def test_run_reservation_enforces_a_per_session_limit(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)
    first = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="First",
        max_pending=2,
    )
    await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-2",
        prompt="Second",
        max_pending=2,
    )

    with raises(RunAdmissionError):
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request-3",
            prompt="Third",
            max_pending=2,
        )
    replay = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="First",
        max_pending=2,
    )

    assert replay.created is False
    assert replay.run.id == first.run.id


async def test_claimed_run_events_are_ordered_and_durable(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)
    reservation = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="Hello",
    )
    claimed = await store.claim_run(
        "tenant/acme",
        reservation.run.id,
        owner_id="worker-1",
        lease_seconds=30,
    )

    first = await store.append_run_event(
        "tenant/acme",
        claimed.id,
        owner_id="worker-1",
        event="part_start",
        data='{"event_kind":"part_start"}',
    )
    second = await store.append_run_event(
        "tenant/acme",
        claimed.id,
        owner_id="worker-1",
        event="part_delta",
        data='{"event_kind":"part_delta"}',
    )
    events = await SQLiteRuntimeStore(database).list_run_events(
        "tenant/acme",
        claimed.id,
        after=0,
    )

    assert claimed.status is RunStatus.RUNNING
    assert claimed.owner_id == "worker-1"
    assert events == [first, second]
    assert [event.sequence for event in events] == [1, 2]


async def test_run_completion_atomically_saves_session_and_result(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)
    reservation = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="Hello",
    )
    run = await store.claim_run(
        "tenant/acme",
        reservation.run.id,
        owner_id="worker-1",
        lease_seconds=30,
    )
    messages = [
        ModelRequest(parts=[UserPromptPart("Hello")]),
        ModelResponse(parts=[TextPart("Hi")]),
    ]
    usage = RunUsage(requests=1, input_tokens=2, output_tokens=1)

    completion = await store.complete_run(
        "tenant/acme",
        run.id,
        owner_id="worker-1",
        expected_session_revision=session.revision,
        messages=messages,
        output={"answer": "Hi"},
        usage=usage,
    )
    reopened = SQLiteRuntimeStore(database)

    assert await reopened.load_session("tenant/acme", session.id) == completion.session
    assert await reopened.load_run("tenant/acme", run.id) == completion.run
    assert completion.run.status is RunStatus.COMPLETED
    assert completion.run.output == {"answer": "Hi"}
    assert completion.run.usage == usage
    assert completion.run.session_revision == completion.session.revision


async def test_failed_run_is_terminal_and_durable(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)
    reservation = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="Hello",
    )
    run = await store.claim_run(
        "tenant/acme",
        reservation.run.id,
        owner_id="worker-1",
        lease_seconds=30,
    )

    failed = await store.fail_run(
        "tenant/acme",
        run.id,
        owner_id="worker-1",
        error_code="agent_run_failed",
        error_detail="Agent run failed",
    )
    reopened = SQLiteRuntimeStore(database)

    assert await reopened.load_run("tenant/acme", run.id) == failed
    assert failed.status is RunStatus.FAILED
    assert failed.error_code == "agent_run_failed"
    assert (await reopened.load_session("tenant/acme", session.id)).messages == []


async def test_expired_run_is_interrupted_and_never_reclaimed(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)
    reservation = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="Hello",
    )
    await store.claim_run(
        "tenant/acme",
        reservation.run.id,
        owner_id="worker-1",
        lease_seconds=0.01,
    )
    await asyncio.sleep(0.02)

    assert await store.interrupt_expired_runs() == 1
    interrupted = await SQLiteRuntimeStore(database).load_run(
        "tenant/acme",
        reservation.run.id,
    )

    assert interrupted.status is RunStatus.INTERRUPTED
    assert interrupted.error_code == "run_interrupted"
    with raises(RunStateError):
        await store.claim_run(
            "tenant/acme",
            interrupted.id,
            owner_id="worker-2",
            lease_seconds=30,
        )


async def test_stale_completion_changes_neither_run_nor_session(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)
    reservation = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="Hello",
    )
    run = await store.claim_run(
        "tenant/acme",
        reservation.run.id,
        owner_id="worker-1",
        lease_seconds=30,
    )
    newer = await store.replace_session(
        "tenant/acme",
        session.id,
        expected_revision=session.revision,
        messages=[ModelRequest(parts=[UserPromptPart("Other")])],
    )

    with raises(SessionConflictError):
        await store.complete_run(
            "tenant/acme",
            run.id,
            owner_id="worker-1",
            expected_session_revision=session.revision,
            messages=[ModelRequest(parts=[UserPromptPart("Lost")])],
            output="lost",
            usage=RunUsage(requests=1),
        )

    assert await store.load_session("tenant/acme", session.id) == newer
    assert (await store.load_run("tenant/acme", run.id)).status is RunStatus.RUNNING
