from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import raises
from yolop_deep import CandidateLimitError, ForkCandidateService, ForkCandidateStatus
from yolop_runtime import RunStatus, Runtime
from yolop_sqlite_session import SQLiteRuntimeStore


def definition_spec() -> AgentSpec:
    return AgentSpec(model="parent:model")


async def setup(tmp_path: Path) -> tuple[Runtime[None], str, str]:
    runtime = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    spec = definition_spec()
    session = await runtime.create_session("tenant/acme", spec=spec, model_id="parent:model")

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "parent checkpoint"

    completion = await runtime.run(
        "tenant/acme",
        session.id,
        "parent work",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        model_id="parent:model",
        deps=None,
        deps_type=type(None),
        idempotency_key="parent",
    )
    return runtime, session.id, completion.run.id


async def test_fork_candidate_isolated_and_restartable(tmp_path: Path) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    spec = definition_spec()

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "candidate result"

    candidate_model = FunctionModel(stream_function=respond)
    service = ForkCandidateService(
        runtime,
        model_for_id=lambda model_id: candidate_model if model_id == "parent:model" else model_id,
        deps_for_candidate=lambda _record: (None, type(None)),
        spec_for_pin=lambda _pin: spec,
    )
    assert "experimental" in service.EXPERIMENTAL_WARNING.lower()
    source_before = await runtime.load_session("tenant/acme", source_session_id)
    handle = await service.start(
        "tenant/acme",
        source_session_id,
        source_run_id,
        spec=spec,
        model_id="parent:model",
        prompt="try an alternative",
        candidate_key="candidate-1",
    )
    restarted = ForkCandidateService(
        runtime,
        model_for_id=lambda model_id: candidate_model if model_id == "parent:model" else model_id,
        deps_for_candidate=lambda _record: (None, type(None)),
        spec_for_pin=lambda _pin: spec,
    )
    completed = await restarted.run(handle)
    source_after = await runtime.load_session("tenant/acme", source_session_id)

    assert completed.status is ForkCandidateStatus.COMPLETED
    assert completed.run_status is RunStatus.COMPLETED
    assert completed.output == "candidate result"
    assert source_after.messages == source_before.messages
    assert completed.candidate_session_id != source_session_id
    candidate_session = await runtime.load_session("tenant/acme", completed.candidate_session_id)
    candidate_run = await runtime.get_run("tenant/acme", completed.candidate_run_id)
    assert candidate_session.pin == source_before.pin
    assert candidate_run.parent_run_id == source_run_id


async def test_candidate_start_is_atomic_under_concurrency(tmp_path: Path) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    service = ForkCandidateService(
        runtime,
        model_for_id=lambda model_id: model_id,
        deps_for_candidate=lambda _record: (None, type(None)),
        spec_for_pin=lambda _pin: definition_spec(),
        max_candidates=1,
    )

    results = await asyncio.gather(
        *(
            service.start(
                "tenant/acme",
                source_session_id,
                source_run_id,
                spec=definition_spec(),
                model_id="parent:model",
                prompt="one",
                candidate_key="candidate-1",
            )
            for _ in range(2)
        )
    )

    assert results[0] == results[1]
    assert len(await runtime.list_sessions("tenant/acme")) == 2
    assert len(await service.list_candidates("tenant/acme", source_session_id, source_run_id)) == 1


async def test_candidate_keys_are_scoped_to_source_run(tmp_path: Path) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    spec = definition_spec()

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "second checkpoint"

    second = await runtime.run(
        "tenant/acme",
        source_session_id,
        "second work",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        model_id="parent:model",
        deps=None,
        deps_type=type(None),
        parent_run_id=source_run_id,
        idempotency_key="second",
    )
    service = ForkCandidateService(
        runtime,
        model_for_id=lambda model_id: model_id,
        deps_for_candidate=lambda _record: (None, type(None)),
        spec_for_pin=lambda _pin: spec,
    )

    first = await service.start(
        "tenant/acme",
        source_session_id,
        source_run_id,
        spec=spec,
        model_id="parent:model",
        prompt="same prompt",
        candidate_key="same-key",
    )
    other = await service.start(
        "tenant/acme",
        source_session_id,
        second.run.id,
        spec=spec,
        model_id="parent:model",
        prompt="same prompt",
        candidate_key="same-key",
    )

    assert first.candidate_session_id != other.candidate_session_id


async def test_candidate_start_is_idempotent_and_limits_are_host_enforced(tmp_path: Path) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    service = ForkCandidateService(
        runtime,
        model_for_id=lambda model_id: model_id,
        deps_for_candidate=lambda _record: (None, type(None)),
        spec_for_pin=lambda _pin: definition_spec(),
        max_candidates=1,
    )
    first = await service.start(
        "tenant/acme",
        source_session_id,
        source_run_id,
        spec=definition_spec(),
        model_id="parent:model",
        prompt="one",
        candidate_key="candidate-1",
    )
    second = await service.start(
        "tenant/acme",
        source_session_id,
        source_run_id,
        spec=definition_spec(),
        model_id="parent:model",
        prompt="one",
        candidate_key="candidate-1",
    )
    assert first == second

    with raises(CandidateLimitError):
        await service.start(
            "tenant/acme",
            source_session_id,
            source_run_id,
            spec=definition_spec(),
            model_id="parent:model",
            prompt="two",
            candidate_key="candidate-2",
        )
