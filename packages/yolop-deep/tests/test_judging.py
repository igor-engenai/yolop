from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import raises
from test_forks import setup
from yolop_deep import (
    CandidateAcceptanceError,
    CandidateJudgeError,
    CandidateJudgeService,
    ForkCandidateService,
)


async def _make_candidates(runtime, source_session_id: str, source_run_id: str):
    spec = AgentSpec(model="parent:model")

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "candidate output"

    model = FunctionModel(stream_function=respond)
    service = ForkCandidateService(
        runtime,
        model_for_id=lambda model_id: model if model_id == "parent:model" else model_id,
        deps_for_candidate=lambda _record: (None, type(None)),
        spec_for_pin=lambda _pin: spec,
    )
    handles = []
    for key in ("candidate-1", "candidate-2"):
        handle = await service.start(
            "tenant/acme",
            source_session_id,
            source_run_id,
            spec=spec,
            model_id="parent:model",
            prompt=f"alternative {key}",
            candidate_key=key,
        )
        handles.append(await service.run(handle))
    return service, handles


async def test_judge_requires_evidence_for_every_candidate(tmp_path: Path) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    candidates, handles = await _make_candidates(
        runtime, source_session_id, source_run_id
    )
    judge = CandidateJudgeService(runtime, candidates=candidates, max_evidence_bytes=180)

    with raises(CandidateJudgeError, match="evidence"):
        await judge.judge(
            "tenant/acme",
            source_session_id,
            source_run_id,
            candidate_keys=[handle.candidate_key for handle in handles],
            evaluator_spec=AgentSpec(model="evaluator:model"),
            evaluator_model=FunctionModel(stream_function=lambda _messages, _info: None),
            evaluator_model_id="evaluator:model",
            deps=None,
            deps_type=type(None),
        )


async def test_judge_persists_a_distinct_verdict_for_each_candidate(tmp_path: Path) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    candidates, handles = await _make_candidates(
        runtime, source_session_id, source_run_id
    )

    async def evaluate(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield (
            '{"judgments":['
            '{"candidate_key":"candidate-1","verdict":"accept","reason":"best"},'
            '{"candidate_key":"candidate-2","verdict":"reject","reason":"worse"}'
            ']}'
        )

    judge = CandidateJudgeService(runtime, candidates=candidates, max_evidence_bytes=1000)
    judgments = await judge.judge(
        "tenant/acme",
        source_session_id,
        source_run_id,
        candidate_keys=[handle.candidate_key for handle in handles],
        evaluator_spec=AgentSpec(model="evaluator:model"),
        evaluator_model=FunctionModel(stream_function=evaluate),
        evaluator_model_id="evaluator:model",
        deps=None,
        deps_type=type(None),
    )

    assert [(item.candidate_key, item.verdict) for item in judgments] == [
        ("candidate-1", "accept"),
        ("candidate-2", "reject"),
    ]


async def test_accept_rejects_a_judgment_from_an_older_revision(tmp_path: Path) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    candidates, _handles = await _make_candidates(runtime, source_session_id, source_run_id)
    candidate = (
        await candidates.list_candidates("tenant/acme", source_session_id, source_run_id)
    )[0]

    async def evaluate(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield (
            '{"judgments":[{"candidate_key":"candidate-1",'
            '"verdict":"accept","reason":"best"}]}'
        )

    judge = CandidateJudgeService(runtime, candidates=candidates)
    await judge.judge(
        "tenant/acme",
        source_session_id,
        source_run_id,
        candidate_keys=[candidate.candidate_key],
        evaluator_spec=AgentSpec(model="evaluator:model"),
        evaluator_model=FunctionModel(stream_function=evaluate),
        evaluator_model_id="evaluator:model",
        deps=None,
        deps_type=type(None),
    )
    changed = await runtime.store.replace_session(
        "tenant/acme",
        source_session_id,
        expected_revision=(await runtime.load_session("tenant/acme", source_session_id)).revision,
        messages=[ModelRequest(parts=[UserPromptPart("new source work")])],
    )

    with raises(CandidateAcceptanceError, match="changed after judging"):
        await judge.accept(
            "tenant/acme",
            source_session_id,
            source_run_id,
            candidate.candidate_key,
            expected_revision=changed.revision,
        )


async def test_judge_reuses_a_completed_evaluator_after_state_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    candidates, handles = await _make_candidates(runtime, source_session_id, source_run_id)
    candidate = handles[0]

    async def evaluate(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield (
            '{"judgments":[{"candidate_key":"candidate-1",'
            '"verdict":"accept","reason":"best"}]}'
        )

    original_append_state = runtime.store.append_state
    fail_once = True

    async def append_state(*args, **kwargs):
        nonlocal fail_once
        if fail_once and kwargs.get("owner_id") == "yolop.deep.fork_judging":
            fail_once = False
            raise RuntimeError("simulated state failure")
        return await original_append_state(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "append_state", append_state)
    judge = CandidateJudgeService(runtime, candidates=candidates)
    with raises(RuntimeError, match="simulated"):
        await judge.judge(
            "tenant/acme",
            source_session_id,
            source_run_id,
            candidate_keys=[candidate.candidate_key],
            evaluator_spec=AgentSpec(model="evaluator:model"),
            evaluator_model=FunctionModel(stream_function=evaluate),
            evaluator_model_id="evaluator:model",
            deps=None,
            deps_type=type(None),
        )

    judgments = await judge.judge(
        "tenant/acme",
        source_session_id,
        source_run_id,
        candidate_keys=[candidate.candidate_key],
        evaluator_spec=AgentSpec(model="evaluator:model"),
        evaluator_model=FunctionModel(stream_function=evaluate),
        evaluator_model_id="evaluator:model",
        deps=None,
        deps_type=type(None),
    )
    evaluator_runs = [
        run
        for run in await runtime.list_runs("tenant/acme")
        if run.initiator == "fork_judge"
    ]
    assert judgments[0].verdict == "accept"
    assert len(evaluator_runs) == 1


async def test_accept_recovers_after_projection_failure(tmp_path: Path, monkeypatch) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    candidates, handles = await _make_candidates(runtime, source_session_id, source_run_id)
    candidate = handles[0]

    async def evaluate(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield (
            '{"judgments":[{"candidate_key":"candidate-1",'
            '"verdict":"accept","reason":"best"}]}'
        )

    judge = CandidateJudgeService(runtime, candidates=candidates)
    await judge.judge(
        "tenant/acme",
        source_session_id,
        source_run_id,
        candidate_keys=[candidate.candidate_key],
        evaluator_spec=AgentSpec(model="evaluator:model"),
        evaluator_model=FunctionModel(stream_function=evaluate),
        evaluator_model_id="evaluator:model",
        deps=None,
        deps_type=type(None),
    )
    source = await runtime.load_session("tenant/acme", source_session_id)
    original_replace = runtime.store.replace_session
    fail_once = True

    async def replace_session(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("simulated projection failure")
        return await original_replace(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "replace_session", replace_session)
    with raises(RuntimeError, match="simulated"):
        await judge.accept(
            "tenant/acme",
            source_session_id,
            source_run_id,
            candidate.candidate_key,
            expected_revision=source.revision,
        )

    accepted = await judge.accept(
        "tenant/acme",
        source_session_id,
        source_run_id,
        candidate.candidate_key,
        expected_revision=source.revision,
    )
    assert accepted.messages == (
        await runtime.get_run("tenant/acme", candidate.candidate_run_id)
    ).active_messages


async def test_judge_persists_evaluator_and_accepts_one_candidate(tmp_path: Path) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path)
    parent_spec = AgentSpec(model="parent:model")

    async def child_respond(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "candidate output"

    child_model = FunctionModel(stream_function=child_respond)
    candidates = ForkCandidateService(
        runtime,
        model_for_id=lambda model_id: child_model if model_id == "parent:model" else model_id,
        deps_for_candidate=lambda _record: (None, type(None)),
        spec_for_pin=lambda _pin: parent_spec,
    )
    candidate = await candidates.start(
        "tenant/acme",
        source_session_id,
        source_run_id,
        spec=parent_spec,
        model_id="parent:model",
        prompt="alternative",
        candidate_key="candidate-1",
    )
    candidate = await candidates.run(candidate)
    source_before = await runtime.load_session("tenant/acme", source_session_id)

    async def evaluate(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield (
            '{"judgments":[{"candidate_key":"candidate-1",'
            '"verdict":"accept","reason":"candidate is better"}]}'
        )

    judge = CandidateJudgeService(
        runtime,
        candidates=candidates,
        max_evidence_bytes=1000,
    )
    judgments = await judge.judge(
        "tenant/acme",
        source_session_id,
        source_run_id,
        candidate_keys=[candidate.candidate_key],
        evaluator_spec=AgentSpec(model="evaluator:model"),
        evaluator_model=FunctionModel(stream_function=evaluate),
        evaluator_model_id="evaluator:model",
        deps=None,
        deps_type=type(None),
    )
    assert judgments[0].verdict == "accept"
    assert judgments[0].evaluator_run_id

    accepted = await judge.accept(
        "tenant/acme",
        source_session_id,
        source_run_id,
        candidate.candidate_key,
        expected_revision=source_before.revision,
    )

    assert accepted.revision != source_before.revision
    assert (
        accepted.messages
        == (await runtime.get_run("tenant/acme", candidate.candidate_run_id)).active_messages
    )

    with raises(CandidateAcceptanceError):
        await judge.accept(
            "tenant/acme",
            source_session_id,
            source_run_id,
            candidate.candidate_key,
            expected_revision=source_before.revision,
        )
