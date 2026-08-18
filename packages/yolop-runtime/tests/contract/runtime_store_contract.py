import asyncio
from pathlib import Path

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RunUsage
from pytest import raises
from yolop_runtime import (
    ExecutionPin,
    IdempotencyConflictError,
    RunBudgetExceededError,
    RunRelation,
    RunReservation,
    RunStateError,
    RunStatus,
    RuntimeBudget,
    RuntimeStore,
    SessionConflictError,
    SessionNotFoundError,
    StateScope,
    StateSequenceConflictError,
)


class RuntimeStoreContract:
    async def make_store(self, database: Path) -> RuntimeStore:
        raise NotImplementedError

    async def test_concurrent_idempotent_reservation_returns_one_run(
        self,
        tmp_path: Path,
    ) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )

        async def reserve() -> RunReservation:
            return await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="request-1",
                prompt="Hello",
            )

        first, second = await asyncio.gather(reserve(), reserve())

        assert first.run.id == second.run.id
        assert sorted((first.created, second.created)) == [False, True]
        assert first.run.status is RunStatus.ACCEPTED

    async def test_idempotency_key_is_bound_to_parent_run(self, tmp_path: Path) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        first_parent = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="first-parent",
            prompt="First",
        )
        second_parent = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="second-parent",
            prompt="Second",
        )
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="continuation",
            prompt="",
            parent_run_id=first_parent.run.id,
            relation=RunRelation.CONTINUATION,
            input_digest="same-input",
        )

        with raises(IdempotencyConflictError):
            await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="continuation",
                prompt="",
                parent_run_id=second_parent.run.id,
                relation=RunRelation.CONTINUATION,
                input_digest="same-input",
            )

    async def test_idempotency_key_is_bound_to_run_relation(self, tmp_path: Path) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        parent = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="parent",
            prompt="Parent",
        )
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="related",
            prompt="Related",
            parent_run_id=parent.run.id,
            relation=RunRelation.CHILD,
        )

        with raises(IdempotencyConflictError):
            await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="related",
                prompt="Related",
                parent_run_id=parent.run.id,
                relation=RunRelation.CONTINUATION,
            )

    async def test_related_runs_can_use_a_different_child_session(self, tmp_path: Path) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        parent_session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="parent:model"),
        )
        child_session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="b" * 64, model_id="child:model"),
        )
        parent = await store.reserve_run(
            "tenant/acme",
            parent_session.id,
            idempotency_key="parent",
            prompt="Parent",
        )

        child = await store.reserve_run(
            "tenant/acme",
            child_session.id,
            idempotency_key="child",
            prompt="Child",
            parent_run_id=parent.run.id,
            relation=RunRelation.CHILD,
        )

        assert child.run.session_id == child_session.id
        assert child.run.parent_run_id == parent.run.id
        assert child.run.root_run_id == parent.run.id

    async def test_idempotency_key_is_bound_to_run_initiator(self, tmp_path: Path) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="request",
            prompt="Run",
            initiator="user",
        )

        with raises(IdempotencyConflictError):
            await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="request",
                prompt="Run",
                initiator="goal-evaluator",
            )

    async def test_sessions_are_isolated_by_namespace(self, tmp_path: Path) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        acme = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        beta = await store.create_session(
            "tenant/beta",
            pin=ExecutionPin(agent_spec_id="b" * 64, model_id="openai:model"),
        )

        assert await store.list_sessions("tenant/acme") == [acme.id]
        assert await store.list_sessions("tenant/beta") == [beta.id]
        with raises(SessionNotFoundError):
            await store.load_session("tenant/beta", acme.id)

    async def test_session_revision_replacement_is_optimistic(self, tmp_path: Path) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )

        results = await asyncio.gather(
            store.replace_session(
                "tenant/acme",
                session.id,
                expected_revision=session.revision,
                messages=[ModelRequest(parts=[UserPromptPart("First")])],
            ),
            store.replace_session(
                "tenant/acme",
                session.id,
                expected_revision=session.revision,
                messages=[ModelRequest(parts=[UserPromptPart("Second")])],
            ),
            return_exceptions=True,
        )

        assert sum(isinstance(result, SessionConflictError) for result in results) == 1
        loaded = await store.load_session("tenant/acme", session.id)
        assert loaded.pin == session.pin
        assert loaded.revision != session.revision
        assert len(loaded.messages) == 1
        assert isinstance(loaded.messages[0], ModelRequest)
        loaded_part = loaded.messages[0].parts[0]
        assert isinstance(loaded_part, UserPromptPart)
        assert loaded_part.content in {"First", "Second"}

    async def test_session_head_and_message_order_are_durable(self, tmp_path: Path) -> None:
        database = tmp_path / "runtime.db"
        store = await self.make_store(database)
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
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
        messages = [
            ModelRequest(parts=[UserPromptPart("Hello")]),
            ModelResponse(parts=[TextPart("Hi")]),
        ]

        completion = await store.complete_run(
            "tenant/acme",
            claimed.id,
            owner_id="worker-1",
            expected_session_revision=session.revision,
            messages=messages,
            output="Hi",
            usage=RunUsage(requests=1, input_tokens=2, output_tokens=1),
        )
        checked_out = await store.checkout_session(
            "tenant/acme",
            session.id,
            claimed.id,
            expected_revision=completion.session.revision,
        )

        assert checked_out.head_run_id == claimed.id
        assert len(checked_out.messages) == 2
        first_message, second_message = checked_out.messages
        assert isinstance(first_message, ModelRequest)
        assert isinstance(second_message, ModelResponse)
        first_part = first_message.parts[0]
        second_part = second_message.parts[0]
        assert isinstance(first_part, UserPromptPart)
        assert isinstance(second_part, TextPart)
        assert first_part.content == "Hello"
        assert second_part.content == "Hi"

    async def test_claims_and_leases_fence_run_ownership(self, tmp_path: Path) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
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
            lease_seconds=1,
        )

        with raises(RunStateError):
            await store.claim_run(
                "tenant/acme",
                reservation.run.id,
                owner_id="worker-2",
                lease_seconds=1,
            )
        with raises(RunStateError):
            await store.renew_run_lease(
                "tenant/acme",
                claimed.id,
                owner_id="worker-2",
                lease_seconds=1,
            )

        renewed = await store.renew_run_lease(
            "tenant/acme",
            claimed.id,
            owner_id="worker-1",
            lease_seconds=30,
        )
        assert renewed.owner_id == "worker-1"
        assert renewed.lease_expires_at is not None
        assert claimed.lease_expires_at is not None
        assert renewed.lease_expires_at > claimed.lease_expires_at

    async def test_shutdown_owned_runs_become_terminal_and_cannot_be_reclaimed(
        self,
        tmp_path: Path,
    ) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
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
            lease_seconds=30,
        )

        assert await store.interrupt_owned_runs("worker-1") == 1
        interrupted = await store.load_run("tenant/acme", reservation.run.id)

        assert interrupted.status is RunStatus.INTERRUPTED
        with raises(RunStateError):
            await store.claim_run(
                "tenant/acme",
                interrupted.id,
                owner_id="worker-2",
                lease_seconds=30,
            )

    async def test_terminal_transitions_are_durable_and_owner_bound(
        self,
        tmp_path: Path,
    ) -> None:
        database = tmp_path / "runtime.db"
        store = await self.make_store(database)
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        failed_reservation = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="failed",
            prompt="Fail",
        )
        failed_claim = await store.claim_run(
            "tenant/acme",
            failed_reservation.run.id,
            owner_id="worker-1",
            lease_seconds=30,
        )

        with raises(RunStateError):
            await store.fail_run(
                "tenant/acme",
                failed_claim.id,
                owner_id="worker-2",
                error_code="wrong-owner",
                error_detail="Wrong owner",
            )
        failed = await store.fail_run(
            "tenant/acme",
            failed_claim.id,
            owner_id="worker-1",
            error_code="agent_run_failed",
            error_detail="Agent run failed",
        )

        completed_reservation = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="completed",
            prompt="Complete",
        )
        completed_claim = await store.claim_run(
            "tenant/acme",
            completed_reservation.run.id,
            owner_id="worker-1",
            lease_seconds=30,
        )
        completed = await store.complete_run(
            "tenant/acme",
            completed_claim.id,
            owner_id="worker-1",
            expected_session_revision=session.revision,
            messages=[],
            output={"answer": "done"},
            usage=RunUsage(requests=1),
        )

        reopened = await self.make_store(database)
        assert failed.status is RunStatus.FAILED
        assert (await reopened.load_run("tenant/acme", failed.id)).status is RunStatus.FAILED
        assert completed.run.status is RunStatus.COMPLETED
        assert completed.run.output == {"answer": "done"}

    async def test_run_events_are_ordered_and_cursor_readable(
        self,
        tmp_path: Path,
    ) -> None:
        database = tmp_path / "runtime.db"
        store = await self.make_store(database)
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
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
            event="tool_start",
            data='{"name":"lookup"}',
        )
        second = await store.append_run_event(
            "tenant/acme",
            claimed.id,
            owner_id="worker-1",
            event="tool_end",
            data='{"ok":true}',
        )
        reopened = await self.make_store(database)

        assert await reopened.list_run_events("tenant/acme", claimed.id) == [first, second]
        assert await reopened.list_run_events("tenant/acme", claimed.id, after=first.sequence) == [
            second
        ]
        assert [event.sequence for event in (first, second)] == [1, 2]

    async def test_plugin_state_is_scoped_and_compare_append_is_bounded(
        self,
        tmp_path: Path,
    ) -> None:
        database = tmp_path / "runtime.db"
        store = await self.make_store(database)
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        first = await store.append_state(
            "tenant/acme",
            owner_id="plugin.counter",
            scope=StateScope.SESSION,
            scope_id=session.id,
            state_kind="counter",
            schema_version=1,
            expected_sequence=0,
            payload={"count": 1},
        )

        results = await asyncio.gather(
            store.append_state(
                "tenant/acme",
                owner_id="plugin.counter",
                scope=StateScope.SESSION,
                scope_id=session.id,
                state_kind="counter",
                schema_version=1,
                expected_sequence=first.sequence,
                payload={"count": 2},
            ),
            store.append_state(
                "tenant/acme",
                owner_id="plugin.counter",
                scope=StateScope.SESSION,
                scope_id=session.id,
                state_kind="counter",
                schema_version=1,
                expected_sequence=first.sequence,
                payload={"count": 3},
            ),
            return_exceptions=True,
        )
        reopened = await self.make_store(database)
        entries = await reopened.read_state(
            "tenant/acme",
            owner_id="plugin.counter",
            scope=StateScope.SESSION,
            scope_id=session.id,
            state_kind="counter",
            schema_version=1,
        )

        assert sum(isinstance(result, StateSequenceConflictError) for result in results) == 1
        assert [entry.sequence for entry in entries] == [1, 2]
        assert entries[0] == first
        assert (
            await reopened.read_state(
                "tenant/beta",
                owner_id="plugin.counter",
                scope=StateScope.SESSION,
                scope_id=session.id,
                state_kind="counter",
                schema_version=1,
            )
            == []
        )

    async def test_run_ancestry_and_root_budget_limits_are_durable(
        self,
        tmp_path: Path,
    ) -> None:
        store = await self.make_store(tmp_path / "runtime.db")
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        root = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="root",
            prompt="Root",
            root_budget=RuntimeBudget(
                request_limit=3,
                child_run_limit=1,
                continuation_limit=1,
            ),
        )
        child = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="child",
            prompt="Child",
            parent_run_id=root.run.id,
            relation=RunRelation.CHILD,
        )
        continuation = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="continuation",
            prompt="Continue",
            parent_run_id=root.run.id,
            relation=RunRelation.CONTINUATION,
        )

        with raises(RunBudgetExceededError):
            await store.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="second-child",
                prompt="Second child",
                parent_run_id=root.run.id,
                relation=RunRelation.CHILD,
            )
        reopened = await self.make_store(tmp_path / "runtime.db")
        loaded_child = await reopened.load_run("tenant/acme", child.run.id)
        loaded_continuation = await reopened.load_run("tenant/acme", continuation.run.id)

        assert loaded_child.root_run_id == root.run.id
        assert loaded_child.relation is RunRelation.CHILD
        assert loaded_continuation.root_run_id == root.run.id
        assert loaded_continuation.relation is RunRelation.CONTINUATION

    async def test_root_budget_usage_survives_restart_and_related_runs(
        self,
        tmp_path: Path,
    ) -> None:
        database = tmp_path / "runtime.db"
        store = await self.make_store(database)
        session = await store.create_session(
            "tenant/acme",
            pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model"),
        )
        root = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="root",
            prompt="Root",
            root_budget=RuntimeBudget(request_limit=2, continuation_limit=1),
        )
        root_claim = await store.claim_run(
            "tenant/acme",
            root.run.id,
            owner_id="worker-root",
            lease_seconds=30,
        )
        await store.complete_run(
            "tenant/acme",
            root_claim.id,
            owner_id="worker-root",
            expected_session_revision=session.revision,
            messages=[],
            output="root",
            usage=RunUsage(requests=1, input_tokens=2, output_tokens=3),
        )
        continuation = await store.reserve_run(
            "tenant/acme",
            session.id,
            idempotency_key="continuation",
            prompt="Continue",
            parent_run_id=root.run.id,
            relation=RunRelation.CONTINUATION,
        )
        continuation_claim = await store.claim_run(
            "tenant/acme",
            continuation.run.id,
            owner_id="worker-continuation",
            lease_seconds=30,
        )
        await store.complete_run(
            "tenant/acme",
            continuation_claim.id,
            owner_id="worker-continuation",
            expected_session_revision=(
                await store.load_session("tenant/acme", session.id)
            ).revision,
            messages=[],
            output="continuation",
            usage=RunUsage(requests=1, input_tokens=4, output_tokens=5),
        )

        reopened = await self.make_store(database)
        budget = await reopened.load_root_budget("tenant/acme", root.run.id)
        assert budget is not None
        assert budget.requests_used == 2
        assert budget.input_tokens_used == 6
        assert budget.output_tokens_used == 8
        assert budget.total_tokens_used == 14
        assert budget.continuations_used == 1
        assert budget.active_runs == 0

        with raises(RunBudgetExceededError):
            await reopened.reserve_run(
                "tenant/acme",
                session.id,
                idempotency_key="too-many",
                prompt="Too many",
                parent_run_id=continuation.run.id,
                relation=RunRelation.CONTINUATION,
            )
