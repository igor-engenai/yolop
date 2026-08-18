from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import AgentSpec
from test_forks import setup
from yolop_deep import (
    ImprovementProposalError,
    ImprovementProposalService,
    ImprovementRevisionConflictError,
    ProposalStatus,
)


async def test_improvement_proposal_requires_review_and_survives_restart(tmp_path: Path) -> None:
    runtime, session_id, source_run_id = await setup(tmp_path)
    target = AgentSpec(name="deep-coding", instructions="Keep the plan current.")
    service = ImprovementProposalService(runtime)
    source_before = await runtime.load_session("tenant/acme", session_id)

    proposal = await service.propose(
        "tenant/acme",
        session_id,
        source_run_id,
        target_spec=target,
        summary="Clarify the verification instruction.",
        patch="Add one sentence that asks for a test result.",
        evidence="The last run did not show a test result.",
    )

    assert proposal.status is ProposalStatus.PENDING
    assert proposal.revision == 1
    assert (await runtime.load_session("tenant/acme", session_id)) == source_before
    assert not hasattr(service, "apply")

    reopened = ImprovementProposalService(runtime)
    stored = await reopened.get("tenant/acme", session_id, source_run_id, proposal.proposal_id)
    assert stored == proposal

    accepted = await reopened.review(
        "tenant/acme",
        session_id,
        source_run_id,
        proposal.proposal_id,
        expected_revision=1,
        status=ProposalStatus.ACCEPTED,
    )
    assert accepted.status is ProposalStatus.ACCEPTED
    assert accepted.revision == 2
    assert (await runtime.load_session("tenant/acme", session_id)) == source_before

    with pytest.raises(ImprovementRevisionConflictError):
        await reopened.review(
            "tenant/acme",
            session_id,
            source_run_id,
            proposal.proposal_id,
            expected_revision=1,
            status=ProposalStatus.REJECTED,
        )


@pytest.mark.parametrize(
    "patch",
    [
        "Set token: sk-secret in the instructions.",
        "Run import subprocess to update the agent.",
        "Change model: provider:model.",
        "Edit /Users/host/private/prompt.yaml.",
    ],
)
async def test_improvement_proposal_rejects_unsafe_content(tmp_path: Path, patch: str) -> None:
    runtime, session_id, source_run_id = await setup(tmp_path)
    service = ImprovementProposalService(runtime)

    with pytest.raises(ImprovementProposalError):
        await service.propose(
            "tenant/acme",
            session_id,
            source_run_id,
            target_spec=AgentSpec(name="deep-coding"),
            summary="Improve the agent.",
            patch=patch,
        )


async def test_improvement_proposal_rejects_cross_namespace_run(tmp_path: Path) -> None:
    runtime, session_id, source_run_id = await setup(tmp_path)
    other = await runtime.create_session(
        "tenant/other", spec=AgentSpec(name="other"), model_id="other:model"
    )
    service = ImprovementProposalService(runtime)

    with pytest.raises(ImprovementProposalError):
        await service.propose(
            "tenant/acme",
            session_id,
            other.id,
            target_spec=AgentSpec(name="deep-coding"),
            summary="Improve the agent.",
            patch="Add a verification sentence.",
        )


async def test_improvement_proposal_enforces_payload_limit(tmp_path: Path) -> None:
    runtime, session_id, source_run_id = await setup(tmp_path)
    service = ImprovementProposalService(runtime, max_patch_chars=10)

    with pytest.raises(ImprovementProposalError):
        await service.propose(
            "tenant/acme",
            session_id,
            source_run_id,
            target_spec=AgentSpec(name="deep-coding"),
            summary="Improve the agent.",
            patch="This patch is too long.",
        )
