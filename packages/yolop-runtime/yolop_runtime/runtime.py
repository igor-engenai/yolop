from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter
from pydantic_ai import AgentSpec, AgentStreamEvent
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import KnownModelName, Model

from yolop import ProviderCatalog, Yolop

from . import (
    ExecutionPin,
    ExecutionScope,
    HostDepsT,
    IdempotencyConflictError,
    RunCompletion,
    RunStateError,
    RunStatus,
    RuntimeDeps,
    RuntimeEventSink,
    RuntimeFollowUpSink,
    RuntimeRunSnapshot,
    RuntimeSessionSnapshot,
    RuntimeStore,
    ensure_session_pin,
)

_STREAM_EVENT_ADAPTER = TypeAdapter(AgentStreamEvent)
_TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED})


class Runtime[HostDepsT]:
    """Host-neutral durable facade above the stateless YoloP kernel."""

    def __init__(
        self,
        *,
        store: RuntimeStore,
        kernel: Yolop | None = None,
        provider_catalog: ProviderCatalog | None = None,
        session_lock_timeout: float = 30.0,
        lease_seconds: float = 60.0,
    ) -> None:
        if session_lock_timeout <= 0:
            raise ValueError("Session lock timeout must be positive")
        if lease_seconds <= 0:
            raise ValueError("Run lease duration must be positive")
        self.store = store
        self.kernel = kernel or Yolop(provider_catalog=provider_catalog)
        self.session_lock_timeout = session_lock_timeout
        self.lease_seconds = lease_seconds

    async def create_session(
        self,
        namespace: str,
        *,
        spec: AgentSpec | dict[str, Any],
        model_id: str,
    ) -> RuntimeSessionSnapshot:
        """Create a session pinned to one immutable AgentSpec and model identity."""
        self.kernel.provider_catalog.validate_spec(spec)
        pin = ExecutionPin.from_spec(spec, model_id=_require_model_id(model_id))
        return await self.store.create_session(namespace, pin=pin)

    async def load_session(self, namespace: str, session_id: str) -> RuntimeSessionSnapshot:
        return await self.store.load_session(namespace, session_id)

    async def list_sessions(self, namespace: str) -> list[str]:
        return await self.store.list_sessions(namespace)

    async def delete_session(self, namespace: str, session_id: str) -> None:
        session = await self.store.load_session(namespace, session_id)
        await self.store.delete_session(
            namespace,
            session_id,
            expected_revision=session.revision,
        )

    async def run(
        self,
        namespace: str,
        session_id: str,
        prompt: str | Sequence[UserContent] | None,
        *,
        spec: AgentSpec | dict[str, Any],
        model: Model | KnownModelName | str | None = None,
        model_id: str | None = None,
        deps: HostDepsT,
        deps_type: type[HostDepsT],
        idempotency_key: str,
        state: Any = None,
        event_sink: RuntimeEventSink | None = None,
        follow_up_sink: RuntimeFollowUpSink | None = None,
        max_pending: int | None = None,
        initiator: str = "user",
    ) -> RunCompletion:
        """Execute one durable run and return its terminal state and session."""
        self.kernel.provider_catalog.validate_spec(spec)
        resolved_model_id = _model_id(spec, model=model, model_id=model_id)
        session = await self.store.load_session(namespace, session_id)
        ensure_session_pin(
            session,
            ExecutionPin.from_spec(spec, model_id=resolved_model_id),
        )
        prompt_text = _prompt_text(prompt)
        parent = (
            await self.store.load_run(namespace, session.head_run_id)
            if session.head_run_id is not None
            else None
        )
        base_full_messages = list(parent.full_messages if parent is not None else session.messages)
        reservation = await self.store.reserve_run(
            namespace,
            session_id,
            idempotency_key=idempotency_key,
            prompt=prompt_text,
            max_pending=max_pending,
            parent_run_id=session.head_run_id,
            root_run_id=parent.root_run_id if parent is not None else None,
            initiator=initiator,
            full_messages=base_full_messages,
            active_messages=session.messages,
        )
        if not reservation.created:
            if reservation.run.status in _TERMINAL_STATUSES:
                return await self._terminal_completion(namespace, reservation.run.id)
            raise IdempotencyConflictError(
                f"Run {reservation.run.id!r} is already active for this idempotency key"
            )

        owner_id = str(uuid4())
        claimed = await self.store.claim_run(
            namespace,
            reservation.run.id,
            owner_id=owner_id,
            lease_seconds=self.lease_seconds,
        )
        scope = ExecutionScope(
            namespace=namespace,
            session_id=session_id,
            run_id=claimed.id,
            parent_run_id=claimed.parent_run_id,
            root_run_id=claimed.root_run_id,
            initiator=initiator,
        )
        runtime_deps = RuntimeDeps(
            scope=scope,
            state=state,
            event_sink=event_sink,
            follow_up_sink=follow_up_sink,
            host=deps,
        )
        current_session = session
        try:
            async with self.store.lock_session(
                namespace,
                session_id,
                timeout=self.session_lock_timeout,
            ):
                current = await self.store.load_session(namespace, session_id)
                current_session = current
                ensure_session_pin(
                    current,
                    ExecutionPin.from_spec(spec, model_id=resolved_model_id),
                )
                result = await self.kernel.execute(
                    spec,
                    prompt,
                    event_stream_handler=_event_handler(
                        self.store,
                        namespace=namespace,
                        run_id=claimed.id,
                        owner_id=owner_id,
                        event_sink=event_sink,
                    ),
                    deps=runtime_deps,
                    deps_type=RuntimeDeps,
                    model=model,
                    message_history=current.messages,
                )
                new_messages = result.new_messages()
                active_messages = result.all_messages()
                completion = await self.store.complete_run(
                    namespace,
                    claimed.id,
                    owner_id=owner_id,
                    expected_session_revision=current.revision,
                    full_messages=[*base_full_messages, *new_messages],
                    active_messages=active_messages,
                    output=result.output,
                    usage=result.usage,
                )
                terminal = await self.store.load_run(namespace, claimed.id)
                return RunCompletion(session=completion.session, run=terminal)
        except RunCancelled as cancelled:
            return await self._store_cancelled_run(
                namespace,
                claimed.id,
                expected_session_revision=current_session.revision,
                base_full_messages=base_full_messages,
                cancelled=cancelled,
            )
        except asyncio.CancelledError as cancelled:
            partial = RunCancelled.from_cancellation(cancelled)
            if partial is not None:
                await self._store_cancelled_run(
                    namespace,
                    claimed.id,
                    expected_session_revision=current_session.revision,
                    base_full_messages=base_full_messages,
                    cancelled=partial,
                )
            else:
                await self.store.cancel_run(
                    namespace,
                    claimed.id,
                    expected_session_revision=current_session.revision,
                    full_messages=base_full_messages,
                    active_messages=current_session.messages,
                )
            raise
        except Exception:
            await self.store.fail_run(
                namespace,
                claimed.id,
                owner_id=owner_id,
                error_code="agent_run_failed",
                error_detail="Agent run failed",
            )
            raise

    async def _store_cancelled_run(
        self,
        namespace: str,
        run_id: str,
        *,
        expected_session_revision: str,
        base_full_messages: Sequence[ModelMessage],
        cancelled: RunCancelled,
    ) -> RunCompletion:
        full_messages = [*base_full_messages, *cancelled.new_messages()]
        active_messages = cancelled.all_messages()
        run = await self.store.cancel_run(
            namespace,
            run_id,
            expected_session_revision=expected_session_revision,
            full_messages=full_messages,
            active_messages=active_messages,
            usage=cancelled.usage,
        )
        session = await self.store.load_session(namespace, run.session_id)
        return RunCompletion(session=session, run=run)

    async def get_run(self, namespace: str, run_id: str) -> RuntimeRunSnapshot:
        return await self.store.load_run(namespace, run_id)

    async def list_runs(
        self,
        namespace: str,
        *,
        session_id: str | None = None,
    ) -> list[RuntimeRunSnapshot]:
        return await self.store.list_runs(namespace, session_id=session_id)

    async def list_run_events(
        self,
        namespace: str,
        run_id: str,
        *,
        after: int = 0,
    ):
        return await self.store.list_run_events(namespace, run_id, after=after)

    async def cancel_run(self, namespace: str, run_id: str) -> RuntimeRunSnapshot:
        return await self.store.cancel_run(namespace, run_id)

    async def checkout(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        *,
        expected_revision: str,
    ) -> RuntimeSessionSnapshot:
        """Select a terminal Run as the Session head without deleting branches."""
        return await self.store.checkout_session(
            namespace,
            session_id,
            run_id,
            expected_revision=expected_revision,
        )

    async def _terminal_completion(self, namespace: str, run_id: str) -> RunCompletion:
        run = await self.store.load_run(namespace, run_id)
        if run.status not in _TERMINAL_STATUSES:
            raise RunStateError(f"Run {run_id!r} is not terminal")
        session = await self.store.load_session(namespace, run.session_id)
        return RunCompletion(session=session, run=run)


def _event_handler(
    store: RuntimeStore,
    *,
    namespace: str,
    run_id: str,
    owner_id: str,
    event_sink: RuntimeEventSink | None,
) -> EventStreamHandler[Any]:
    async def handle(_context: Any, events: Any) -> None:
        async for event in events:
            if event_sink is not None:
                await event_sink.emit(event)
            await store.append_run_event(
                namespace,
                run_id,
                owner_id=owner_id,
                event=event.event_kind,
                data=_STREAM_EVENT_ADAPTER.dump_json(event).decode(),
            )

    return handle


def _model_id(
    spec: AgentSpec | dict[str, Any],
    *,
    model: Model | KnownModelName | str | None,
    model_id: str | None,
) -> str:
    if model_id is not None:
        return _require_model_id(model_id)
    if isinstance(model, str):
        return _require_model_id(model)
    if isinstance(spec, AgentSpec):
        value = spec.model
    else:
        value = spec.get("model")
    if isinstance(value, str):
        return _require_model_id(value)
    raise ValueError("model_id is required when the resolved model is not a string reference")


def _require_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    return model_id


def _prompt_text(prompt: str | Sequence[UserContent] | None) -> str:
    if isinstance(prompt, str):
        return prompt
    if prompt is None:
        return ""
    return "[structured prompt]"
