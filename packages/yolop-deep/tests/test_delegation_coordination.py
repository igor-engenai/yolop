from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.planning import PlanItem, TaskStatus
from pytest import raises
from yolop_deep import (
    ChannelAuthorizationError,
    DelegatedTaskCoordinator,
    PlanDependencyError,
    deep_delegation_aliases,
)
from yolop_delegation import BackgroundDelegationService, DelegateCatalog, DelegateDefinition
from yolop_runtime import Runtime
from yolop_sqlite_session import SQLiteRuntimeStore


def definition(alias: str, *, model_id: str) -> DelegateDefinition:
    return DelegateDefinition.from_spec(
        alias=alias,
        version="2026-08-18",
        spec=AgentSpec(name=f"{alias}-agent", instructions="Return concise notes."),
        model_id=model_id,
    )


def make_catalog() -> DelegateCatalog:
    return DelegateCatalog(
        {
            "tenant/acme": [
                definition(alias="research", model_id="child:model"),
                definition(alias="review", model_id="child:model"),
            ]
        }
    )


async def setup(
    tmp_path: Path,
) -> tuple[Runtime[None], DelegatedTaskCoordinator, AgentSpec, str, str]:
    runtime = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    parent_spec = AgentSpec(
        model="parent:model",
        metadata={"delegation": {"delegates": [{"alias": "research"}, {"alias": "review"}]}},
    )
    session = await runtime.create_session("tenant/acme", spec=parent_spec, model_id="parent:model")
    parent = await runtime.store.reserve_run(
        "tenant/acme",
        session.id,
        idempotency_key="parent",
        prompt="coordinate",
    )

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "child result"

    child_model = FunctionModel(stream_function=respond)
    background = BackgroundDelegationService(
        runtime,
        catalog=make_catalog(),
        model_for_id=lambda model_id: child_model if model_id == "child:model" else model_id,
        deps_for_request=lambda _request: (None, type(None)),
    )
    return (
        runtime,
        DelegatedTaskCoordinator(runtime, background=background, catalog=make_catalog()),
        parent_spec,
        session.id,
        parent.run.id,
    )


async def test_assign_start_restart_and_complete_one_plan_item(tmp_path: Path) -> None:
    runtime, coordinator, spec, session_id, parent_run_id = await setup(tmp_path)
    await coordinator.add_plan_item(
        "tenant/acme",
        session_id,
        PlanItem(id="research", content="Research the problem"),
        run_id=parent_run_id,
    )
    assignment = await coordinator.assign(
        "tenant/acme",
        session_id,
        "research",
        alias="research",
        parent_spec=spec,
        run_id=parent_run_id,
    )
    handle = await coordinator.start_available(
        "tenant/acme",
        session_id,
        "research",
        parent_run_id=parent_run_id,
        parent_spec=spec,
    )

    assert assignment.alias == "research"
    assert handle.child_run_id is not None
    assert (
        await coordinator.plan_item("tenant/acme", session_id, "research", run_id=parent_run_id)
    ).status is TaskStatus.in_progress

    restarted = DelegatedTaskCoordinator(
        runtime,
        background=coordinator.background,
        catalog=make_catalog(),
    )
    await restarted.background.run_worker(handle)
    completed = await restarted.complete_item(
        "tenant/acme",
        session_id,
        "research",
        run_id=parent_run_id,
    )

    assert completed.status is TaskStatus.completed
    assert await runtime.get_run("tenant/acme", handle.child_run_id)


def test_deep_preset_uses_fixed_aliases_without_a_team_entity() -> None:
    assert deep_delegation_aliases() == ("research", "review")
    assert "Team" not in DelegatedTaskCoordinator.__name__


async def test_dependencies_block_and_channel_authorization_is_scoped(tmp_path: Path) -> None:
    runtime, coordinator, spec, session_id, parent_run_id = await setup(tmp_path)
    await coordinator.add_plan_item(
        "tenant/acme",
        session_id,
        PlanItem(id="first", content="First"),
        run_id=parent_run_id,
    )
    await coordinator.add_plan_item(
        "tenant/acme",
        session_id,
        PlanItem(id="second", content="Second", depends_on=["first"]),
        run_id=parent_run_id,
    )
    await coordinator.assign(
        "tenant/acme",
        session_id,
        "second",
        alias="review",
        parent_spec=spec,
        run_id=parent_run_id,
    )

    with raises(PlanDependencyError):
        await coordinator.start_available(
            "tenant/acme",
            session_id,
            "second",
            parent_run_id=parent_run_id,
            parent_spec=spec,
        )

    await coordinator.assign(
        "tenant/acme",
        session_id,
        "first",
        alias="research",
        parent_spec=spec,
        run_id=parent_run_id,
    )
    first_handle = await coordinator.start_available(
        "tenant/acme",
        session_id,
        "first",
        parent_run_id=parent_run_id,
        parent_spec=spec,
    )
    await coordinator.publish_message(
        "tenant/acme",
        session_id,
        sender_run_id=first_handle.child_run_id,
        recipient_alias="review",
        content="first result",
        run_id=parent_run_id,
    )
    messages = await coordinator.messages("tenant/acme", session_id, run_id=parent_run_id)
    assert messages[0].sequence == 1
    assert messages[0].content == "first result"

    with raises(ChannelAuthorizationError):
        await coordinator.publish_message(
            "tenant/acme",
            session_id,
            sender_run_id="00000000-0000-4000-8000-000000000999",
            recipient_alias="review",
            content="forged",
            run_id=parent_run_id,
        )
