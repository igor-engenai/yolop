from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import raises
from test_forks import setup
from yolop_deep import CandidateAcceptanceError, CandidateJudgeService, ForkCandidateService


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
        yield '{"verdict":"accept","reason":"candidate is better"}'

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
