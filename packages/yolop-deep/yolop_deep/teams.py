"""Durable planning/delegation composition without a Team execution entity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import AgentSpec
from pydantic_ai_harness.planning import PlanItem, TaskStatus
from yolop_delegation import (
    BackgroundDelegationService,
    BackgroundTaskHandle,
    BackgroundTaskStatus,
    DelegateCatalog,
    DelegatePin,
)
from yolop_runtime import RunStatus, Runtime, ScopedStateContext

from . import SessionPlanStore

_COORDINATOR_OWNER = "yolop.deep.delegation"
_COORDINATOR_STATE_KIND = "coordination"
_COORDINATOR_SCHEMA_VERSION = 1


class PlanDependencyError(ValueError):
    """A plan item cannot start because a dependency is not completed."""

    code = "plan_dependency_blocked"


class AssignmentConflictError(ValueError):
    """A plan item has an incompatible delegate assignment."""

    code = "plan_assignment_conflict"


class ChannelAuthorizationError(ValueError):
    """A channel sender or recipient is not authorized by plan assignments."""

    code = "plan_channel_unauthorized"


class TaskAssignment(BaseModel):
    """Durable assignment metadata; child Run state remains canonical."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1, max_length=255)
    alias: str = Field(min_length=1, max_length=64)
    operation_key: str = Field(min_length=1, max_length=255)
    child_session_id: str | None = None
    child_run_id: str | None = None
    pin: DelegatePin | None = None

    @classmethod
    def pending(cls, item_id: str, alias: str) -> TaskAssignment:
        return cls(
            item_id=item_id,
            alias=alias,
            operation_key=f"plan:{item_id}",
        )

    @classmethod
    def from_handle(cls, item_id: str, alias: str, handle: BackgroundTaskHandle) -> TaskAssignment:
        return cls(
            item_id=item_id,
            alias=alias,
            operation_key=handle.operation_key,
            child_session_id=handle.child_session_id,
            child_run_id=handle.child_run_id,
            pin=handle.pin,
        )


class ChannelMessage(BaseModel):
    """One ordered, bounded message in the Session-scoped coordination log."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(gt=0)
    sender_run_id: str
    sender_alias: str
    recipient_alias: str | None = None
    content: str = Field(min_length=1, max_length=16_384)


class _CoordinatorSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assignments: dict[str, TaskAssignment] = Field(default_factory=dict)
    messages: list[ChannelMessage] = Field(default_factory=list)


class _CoordinatorState:
    def __init__(self, runtime: Runtime[Any], namespace: str, session_id: str, run_id: str) -> None:
        self._state = ScopedStateContext(
            store=runtime.store,
            namespace=namespace,
            session_id=session_id,
            run_id=run_id,
        ).for_session(_COORDINATOR_OWNER)

    async def read(self) -> _CoordinatorSnapshot:
        entries = await self._state.read(
            _COORDINATOR_STATE_KIND,
            schema_version=_COORDINATOR_SCHEMA_VERSION,
        )
        if not entries:
            return _CoordinatorSnapshot()
        payload = entries[-1].payload
        try:
            return _CoordinatorSnapshot.model_validate(payload)
        except (TypeError, ValueError) as error:
            raise AssignmentConflictError("Stored coordination state is invalid") from error

    async def write(self, snapshot: _CoordinatorSnapshot) -> None:
        entries = await self._state.read(
            _COORDINATOR_STATE_KIND,
            schema_version=_COORDINATOR_SCHEMA_VERSION,
        )
        expected_sequence = entries[-1].sequence if entries else 0
        await self._state.append(
            _COORDINATOR_STATE_KIND,
            snapshot.model_dump(mode="json"),
            schema_version=_COORDINATOR_SCHEMA_VERSION,
            expected_sequence=expected_sequence,
        )


class DelegatedTaskCoordinator:
    """Compose SessionPlanStore and background Runs as one host operation facade."""

    def __init__(
        self,
        runtime: Runtime[Any],
        *,
        background: BackgroundDelegationService,
        catalog: DelegateCatalog,
        max_message_chars: int = 16_384,
    ) -> None:
        if isinstance(max_message_chars, bool) or max_message_chars < 1:
            raise ValueError("Coordinator max_message_chars must be positive")
        self.runtime = runtime
        self.background = background
        self.catalog = catalog
        self.max_message_chars = max_message_chars

    async def add_plan_item(
        self,
        namespace: str,
        session_id: str,
        item: PlanItem,
        *,
        run_id: str,
    ) -> PlanItem:
        async with self.runtime.store.lock_session(namespace, session_id, timeout=30):
            return await self._plan(namespace, session_id, run_id).add_item(item)

    async def plan_item(
        self,
        namespace: str,
        session_id: str,
        item_id: str,
        *,
        run_id: str,
    ) -> PlanItem:
        item = await self._plan(namespace, session_id, run_id).get_item(item_id)
        if item is None:
            raise ValueError(f"Plan item {item_id!r} does not exist")
        return item

    async def assign(
        self,
        namespace: str,
        session_id: str,
        item_id: str,
        *,
        alias: str,
        parent_spec: AgentSpec | Mapping[str, Any],
        run_id: str,
    ) -> TaskAssignment:
        self.catalog.resolve_for_invocation(namespace, parent_spec, alias)
        await self.plan_item(namespace, session_id, item_id, run_id=run_id)
        state = self._state(namespace, session_id, run_id)
        async with self.runtime.store.lock_session(namespace, session_id, timeout=30):
            snapshot = await state.read()
            existing = snapshot.assignments.get(item_id)
            if existing is not None:
                if existing.alias != alias:
                    raise AssignmentConflictError(
                        f"Plan item {item_id!r} is assigned to {existing.alias!r}"
                    )
                return existing
            assignment = TaskAssignment.pending(item_id, alias)
            await state.write(
                snapshot.model_copy(
                    update={"assignments": {**snapshot.assignments, item_id: assignment}}
                )
            )
            return assignment

    async def start_available(
        self,
        namespace: str,
        session_id: str,
        item_id: str,
        *,
        parent_run_id: str,
        parent_spec: AgentSpec | Mapping[str, Any],
    ) -> BackgroundTaskHandle:
        plan = self._plan(namespace, session_id, parent_run_id)
        item = await plan.get_item(item_id)
        if item is None:
            raise ValueError(f"Plan item {item_id!r} does not exist")
        items = await plan.get_items()
        blocked = {
            dependency.id
            for dependency in items
            if dependency.id in item.depends_on and dependency.status is not TaskStatus.completed
        }
        if blocked:
            raise PlanDependencyError(
                f"Plan item {item_id!r} depends on incomplete items: {', '.join(sorted(blocked))}"
            )
        state = self._state(namespace, session_id, parent_run_id)
        snapshot = await state.read()
        assignment = snapshot.assignments.get(item_id)
        if assignment is None:
            raise AssignmentConflictError(f"Plan item {item_id!r} has no delegate assignment")
        if assignment.child_run_id is not None:
            return await self.background.inspect(
                _handle_from_assignment(assignment, namespace, session_id, parent_run_id)
            )

        handle = await self.background.start(
            namespace,
            session_id,
            parent_run_id,
            parent_spec=parent_spec,
            alias=assignment.alias,
            task=item.content,
            operation_key=assignment.operation_key,
        )
        async with self.runtime.store.lock_session(namespace, session_id, timeout=30):
            current = await state.read()
            current_assignment = current.assignments.get(item_id)
            if current_assignment is not None and current_assignment.child_run_id is not None:
                return await self.background.inspect(
                    _handle_from_assignment(
                        current_assignment, namespace, session_id, parent_run_id
                    )
                )
            updated = TaskAssignment.from_handle(item_id, assignment.alias, handle)
            await state.write(
                current.model_copy(
                    update={"assignments": {**current.assignments, item_id: updated}}
                )
            )
            current_item = await plan.get_item(item_id)
            if current_item is not None and current_item.status is TaskStatus.pending:
                await plan.update_item(
                    item_id,
                    status=TaskStatus.in_progress,
                    active_form=f"Delegating {current_item.content}",
                )
        return handle

    async def complete_item(
        self,
        namespace: str,
        session_id: str,
        item_id: str,
        *,
        run_id: str,
    ) -> PlanItem:
        assignment = await self._assignment(namespace, session_id, item_id, run_id)
        if assignment.child_run_id is None:
            raise AssignmentConflictError(f"Plan item {item_id!r} has not started")
        handle = _handle_from_assignment(assignment, namespace, session_id, run_id)
        view = await self.background.inspect(handle)
        if view.run_status not in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
        }:
            raise AssignmentConflictError(f"Plan item {item_id!r} is still running")
        status = (
            TaskStatus.completed if view.run_status is RunStatus.COMPLETED else TaskStatus.cancelled
        )
        async with self.runtime.store.lock_session(namespace, session_id, timeout=30):
            item = await self._plan(namespace, session_id, run_id).get_item(item_id)
            if item is None:
                raise ValueError(f"Plan item {item_id!r} does not exist")
            if item.status is not status:
                updated = await self._plan(namespace, session_id, run_id).update_item(
                    item_id,
                    status=status,
                    active_form="",
                )
                assert updated is not None
                return updated
            return item

    async def publish_message(
        self,
        namespace: str,
        session_id: str,
        *,
        sender_run_id: str,
        recipient_alias: str | None,
        content: str,
        run_id: str,
    ) -> ChannelMessage:
        if not content.strip() or len(content) > self.max_message_chars:
            raise ChannelAuthorizationError("Channel message is empty or exceeds the host limit")
        state = self._state(namespace, session_id, run_id)
        async with self.runtime.store.lock_session(namespace, session_id, timeout=30):
            snapshot = await state.read()
            sender = next(
                (
                    assignment
                    for assignment in snapshot.assignments.values()
                    if assignment.child_run_id == sender_run_id
                ),
                None,
            )
            if sender is None:
                raise ChannelAuthorizationError("Channel sender is not an assigned child")
            if recipient_alias is not None and recipient_alias not in {
                assignment.alias for assignment in snapshot.assignments.values()
            }:
                raise ChannelAuthorizationError("Channel recipient is not an assigned delegate")
            message = ChannelMessage(
                sequence=len(snapshot.messages) + 1,
                sender_run_id=sender_run_id,
                sender_alias=sender.alias,
                recipient_alias=recipient_alias,
                content=content,
            )
            await state.write(
                snapshot.model_copy(update={"messages": [*snapshot.messages, message]})
            )
            return message

    async def messages(
        self,
        namespace: str,
        session_id: str,
        *,
        run_id: str,
    ) -> list[ChannelMessage]:
        return list((await self._state(namespace, session_id, run_id).read()).messages)

    async def _assignment(
        self, namespace: str, session_id: str, item_id: str, run_id: str
    ) -> TaskAssignment:
        assignment = (await self._state(namespace, session_id, run_id).read()).assignments.get(
            item_id
        )
        if assignment is None:
            raise AssignmentConflictError(f"Plan item {item_id!r} has no assignment")
        return assignment

    def _state(self, namespace: str, session_id: str, run_id: str) -> _CoordinatorState:
        return _CoordinatorState(self.runtime, namespace, session_id, run_id)

    def _plan(self, namespace: str, session_id: str, run_id: str) -> SessionPlanStore:
        return SessionPlanStore(
            ScopedStateContext(
                store=self.runtime.store,
                namespace=namespace,
                session_id=session_id,
                run_id=run_id,
            )
        )


def _handle_from_assignment(
    assignment: TaskAssignment,
    namespace: str,
    parent_session_id: str,
    parent_run_id: str,
) -> BackgroundTaskHandle:
    assert assignment.child_session_id is not None
    assert assignment.child_run_id is not None
    assert assignment.pin is not None
    return BackgroundTaskHandle(
        namespace=namespace,
        operation_key=assignment.operation_key,
        parent_session_id=parent_session_id,
        parent_run_id=parent_run_id,
        child_session_id=assignment.child_session_id,
        child_run_id=assignment.child_run_id,
        pin=assignment.pin,
        status=BackgroundTaskStatus.ACCEPTED,
        run_status=RunStatus.ACCEPTED,
    )


def deep_delegation_aliases() -> tuple[str, str]:
    """Return the fixed aliases used by the explicit deep coordination preset."""
    return ("research", "review")


__all__ = [
    "AssignmentConflictError",
    "ChannelAuthorizationError",
    "ChannelMessage",
    "DelegatedTaskCoordinator",
    "PlanDependencyError",
    "TaskAssignment",
    "deep_delegation_aliases",
]
