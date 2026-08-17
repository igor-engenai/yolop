import asyncio
from uuid import UUID

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RunUsage
from pytest import raises
from yolop_runtime import (
    ExecutionPin,
    IdempotencyConflictError,
    RunAdmissionError,
    RunBudgetExceededError,
    RunRelation,
    RunStateError,
    RunStatus,
    RuntimeBudget,
    SessionConflictError,
)
from yolop_sqlite_session import SQLiteRuntimeStore


async def test_root_budget_accounts_related_runs_and_survives_restart(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    session = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
    )
    budget = RuntimeBudget(request_limit=2, continuation_limit=1, child_run_limit=1)
    root = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="root",
        prompt="Root",
        root_budget=budget,
    )
    claimed_root = await store.claim_run(
        "tenant/acme",
        root.run.id,
        owner_id="worker-root",
        lease_seconds=30,
    )
    await store.complete_run(
        "tenant/acme",
        claimed_root.id,
        owner_id="worker-root",
        expected_session_revision=session.revision,
        messages=[],
        output="root",
        usage=RunUsage(requests=1, input_tokens=3, output_tokens=2),
    )

    continuation = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="continuation",
        prompt="Continue",
        parent_run_id=root.run.id,
        root_run_id=root.run.id,
        relation=RunRelation.CONTINUATION,
    )
    claimed_continuation = await store.claim_run(
        "tenant/acme",
        continuation.run.id,
        owner_id="worker-continuation",
        lease_seconds=30,
    )
    completed = await store.complete_run(
        "tenant/acme",
        claimed_continuation.id,
        owner_id="worker-continuation",
        expected_session_revision=(await store.load_session("tenant/acme", session.id)).revision,
        messages=[],
        output="continuation",
        usage=RunUsage(requests=1, input_tokens=4, output_tokens=5),
    )

    reopened = SQLiteRuntimeStore(database)
    account = await reopened.load_root_budget("tenant/acme", root.run.id)
    assert account is not None
    assert account.requests_used == 2
    assert account.input_tokens_used == 7
    assert account.output_tokens_used == 7
    assert account.total_tokens_used == 14
    assert account.continuations_used == 1
    assert account.active_runs == 0
    assert completed.run.root_run_id == root.run.id

    with raises(RunBudgetExceededError):
        await reopened.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="too-many",
            prompt="Too many",
            parent_run_id=continuation.run.id,
            root_run_id=root.run.id,
            relation=RunRelation.CONTINUATION,
        )


async def test_root_budget_shares_child_limit_with_descendants(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
    )
    root = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="root",
        prompt="Root",
        root_budget=RuntimeBudget(child_run_limit=1),
    )
    child = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="child",
        prompt="Child",
        parent_run_id=root.run.id,
        relation=RunRelation.CHILD,
    )

    assert child.run.relation is RunRelation.CHILD
    with raises(RunBudgetExceededError):
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="second-child",
            prompt="Second child",
            parent_run_id=root.run.id,
            relation=RunRelation.CHILD,
        )


async def test_root_budgets_are_isolated_for_concurrent_sessions(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    first_session = await store.create_session("tenant/acme", pin=pin)
    second_session = await store.create_session("tenant/acme", pin=pin)
    first = await store.reserve_run(
        "tenant/acme",
        first_session.id,
        idempotency_key="first",
        prompt="First",
        root_budget=RuntimeBudget(request_limit=1),
    )
    second = await store.reserve_run(
        "tenant/acme",
        second_session.id,
        idempotency_key="second",
        prompt="Second",
        root_budget=RuntimeBudget(request_limit=1),
    )
    first_claimed = await store.claim_run(
        "tenant/acme", first.run.id, owner_id="worker-1", lease_seconds=30
    )
    second_claimed = await store.claim_run(
        "tenant/acme", second.run.id, owner_id="worker-2", lease_seconds=30
    )
    await store.complete_run(
        "tenant/acme",
        first_claimed.id,
        owner_id="worker-1",
        expected_session_revision=first_session.revision,
        messages=[],
        output="first",
        usage=RunUsage(requests=1),
    )
    await store.complete_run(
        "tenant/acme",
        second_claimed.id,
        owner_id="worker-2",
        expected_session_revision=second_session.revision,
        messages=[],
        output="second",
        usage=RunUsage(requests=1),
    )

    first_budget = await store.load_root_budget("tenant/acme", first.run.id)
    second_budget = await store.load_root_budget("tenant/acme", second.run.id)
    assert first_budget is not None and first_budget.requests_used == 1
    assert second_budget is not None and second_budget.requests_used == 1


async def test_cancelled_root_stops_future_related_runs(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    session = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
    )
    root = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="root",
        prompt="Root",
        root_budget=RuntimeBudget(continuation_limit=2),
    )
    await store.cancel_run("tenant/acme", root.run.id)

    account = await store.load_root_budget("tenant/acme", root.run.id)
    assert account is not None and account.stopped is True
    with raises(RunBudgetExceededError):
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="continuation",
            prompt="Continuation",
            parent_run_id=root.run.id,
            relation=RunRelation.CONTINUATION,
        )


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


async def test_runs_can_be_listed_and_cancelled(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    pin = ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model")
    session = await store.create_session("tenant/acme", pin=pin)
    first = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-1",
        prompt="First",
    )
    second = await store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="request-2",
        prompt="Second",
    )

    listed = await store.list_runs("tenant/acme", session_id=session.id)
    cancelled = await store.cancel_run("tenant/acme", first.run.id)

    assert [run.id for run in listed] == [first.run.id, second.run.id]
    assert cancelled.status is RunStatus.INTERRUPTED
    assert (await store.load_session("tenant/acme", session.id)).head_run_id == first.run.id
    assert (await store.cancel_run("tenant/acme", first.run.id)) == cancelled


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
