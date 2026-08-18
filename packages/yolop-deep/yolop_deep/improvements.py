"""Optional, reviewed agent-improvement proposals with no automatic application."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import AgentSpec
from yolop_runtime import RunNotFoundError, Runtime, ScopedStateContext, agent_spec_digest

_IMPROVEMENT_OWNER = "yolop.deep.improvements"
_IMPROVEMENT_STATE_KIND = "proposals"
_IMPROVEMENT_SCHEMA_VERSION = 1
_SENSITIVE_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|authorization|bearer|password|secret|token)\s*[:=]"
)
_FORBIDDEN_CODE_PATTERN = re.compile(r"(?i)\b(?:import|exec|eval|subprocess|def|class)\b")
_FORBIDDEN_PATH_PATTERN = re.compile(r"(?:^|[\s=:])(?:/|[A-Za-z]:[\\/]|\.\.)")
_FORBIDDEN_POLICY_PATTERN = re.compile(r"(?i)\b(?:model|provider|capabilities?)\s*[:=]")


class ImprovementProposalError(ValueError):
    """An improvement proposal violates the host review policy."""

    code = "improvement_proposal_error"


class ImprovementRevisionConflictError(ImprovementProposalError):
    """A proposal was reviewed from a stale revision."""

    code = "improvement_revision_conflict"


class ImprovementProposalNotFoundError(ImprovementProposalError):
    """A proposal is not available in the requested Session."""

    code = "improvement_proposal_not_found"


class ProposalStatus(StrEnum):
    """Explicit host review lifecycle."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ImprovementProposal(BaseModel):
    """Durable review data, never an executable patch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    namespace: str
    session_id: str
    source_run_id: str
    target_agent_spec_id: str
    summary: str = Field(min_length=1, max_length=2048)
    patch: str = Field(min_length=1, max_length=16_384)
    evidence: str = Field(default="", max_length=32_768)
    status: ProposalStatus = ProposalStatus.PENDING
    revision: int = Field(default=1, ge=1)


class _ImprovementSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposals: dict[str, ImprovementProposal] = Field(default_factory=dict)


class _ImprovementState:
    def __init__(self, runtime: Runtime[Any], namespace: str, session_id: str, run_id: str) -> None:
        self._state = ScopedStateContext(
            store=runtime.store,
            namespace=namespace,
            session_id=session_id,
            run_id=run_id,
        ).for_session(_IMPROVEMENT_OWNER)

    async def read(self) -> _ImprovementSnapshot:
        entries = await self._state.read(
            _IMPROVEMENT_STATE_KIND,
            schema_version=_IMPROVEMENT_SCHEMA_VERSION,
        )
        proposals: dict[str, ImprovementProposal] = {}
        try:
            for entry in entries:
                payload = entry.payload
                if not isinstance(payload, dict):
                    raise ValueError("proposal event is not an object")
                if "proposal" in payload:
                    proposal = ImprovementProposal.model_validate(payload["proposal"])
                    proposals[proposal.proposal_id] = proposal
                elif "proposals" in payload:
                    legacy = _ImprovementSnapshot.model_validate(payload)
                    proposals.update(legacy.proposals)
                else:
                    raise ValueError("proposal event has no proposal")
        except (TypeError, ValueError) as error:
            raise ImprovementProposalError("Stored improvement state is invalid") from error
        return _ImprovementSnapshot(proposals=proposals)

    async def write(self, proposal: ImprovementProposal) -> None:
        entries = await self._state.read(
            _IMPROVEMENT_STATE_KIND,
            schema_version=_IMPROVEMENT_SCHEMA_VERSION,
        )
        expected_sequence = entries[-1].sequence if entries else 0
        await self._state.append(
            _IMPROVEMENT_STATE_KIND,
            {"proposal": proposal.model_dump(mode="json")},
            schema_version=_IMPROVEMENT_SCHEMA_VERSION,
            expected_sequence=expected_sequence,
        )


class ImprovementProposalService:
    """Persist bounded proposals and require explicit review decisions."""

    def __init__(
        self,
        runtime: Runtime[Any],
        *,
        max_summary_chars: int = 2048,
        max_patch_chars: int = 16_384,
        max_evidence_chars: int = 32_768,
    ) -> None:
        self.runtime = runtime
        self.max_summary_chars = max_summary_chars
        self.max_patch_chars = max_patch_chars
        self.max_evidence_chars = max_evidence_chars

    async def propose(
        self,
        namespace: str,
        session_id: str,
        source_run_id: str,
        *,
        target_spec: AgentSpec | Mapping[str, Any],
        summary: str,
        patch: str,
        evidence: str = "",
    ) -> ImprovementProposal:
        _bounded(summary, self.max_summary_chars, "Improvement summary")
        _bounded(patch, self.max_patch_chars, "Improvement patch")
        if evidence:
            _bounded(evidence, self.max_evidence_chars, "Improvement evidence")
        _validate_content(summary, "Improvement summary")
        _validate_content(patch, "Improvement patch")
        if evidence:
            _validate_content(evidence, "Improvement evidence")
        await self.runtime.load_session(namespace, session_id)
        try:
            source = await self.runtime.get_run(namespace, source_run_id)
        except RunNotFoundError as error:
            raise ImprovementProposalError(
                "Source Run is not in the requested namespace"
            ) from error
        if source.session_id != session_id:
            raise ImprovementProposalError("Source Run does not belong to the Session")
        spec = (
            target_spec
            if isinstance(target_spec, AgentSpec)
            else AgentSpec.model_validate(target_spec)
        )
        proposal = ImprovementProposal(
            proposal_id=str(uuid4()),
            namespace=namespace,
            session_id=session_id,
            source_run_id=source_run_id,
            target_agent_spec_id=agent_spec_digest(spec),
            summary=summary,
            patch=patch,
            evidence=evidence,
        )
        state = _ImprovementState(self.runtime, namespace, session_id, source_run_id)
        async with self.runtime.store.lock_session(namespace, session_id, timeout=30):
            await state.write(proposal)
        return proposal

    async def get(
        self, namespace: str, session_id: str, source_run_id: str, proposal_id: str
    ) -> ImprovementProposal:
        proposal = (
            await _ImprovementState(self.runtime, namespace, session_id, source_run_id).read()
        ).proposals.get(proposal_id)
        if proposal is None:
            raise ImprovementProposalNotFoundError("Improvement proposal is not available")
        return proposal

    async def list_proposals(
        self, namespace: str, session_id: str, source_run_id: str
    ) -> list[ImprovementProposal]:
        return list(
            (
                await _ImprovementState(self.runtime, namespace, session_id, source_run_id).read()
            ).proposals.values()
        )

    async def review(
        self,
        namespace: str,
        session_id: str,
        source_run_id: str,
        proposal_id: str,
        *,
        expected_revision: int,
        status: ProposalStatus,
    ) -> ImprovementProposal:
        if status is ProposalStatus.PENDING:
            raise ImprovementProposalError("Review must decide accepted or rejected")
        state = _ImprovementState(self.runtime, namespace, session_id, source_run_id)
        async with self.runtime.store.lock_session(namespace, session_id, timeout=30):
            snapshot = await state.read()
            current = snapshot.proposals.get(proposal_id)
            if current is None:
                raise ImprovementProposalNotFoundError("Improvement proposal is not available")
            if (
                current.revision != expected_revision
                or current.status is not ProposalStatus.PENDING
            ):
                raise ImprovementRevisionConflictError("Improvement proposal revision is stale")
            reviewed = current.model_copy(
                update={"status": status, "revision": current.revision + 1}
            )
            await state.write(reviewed)
            return reviewed


def _bounded(value: str, limit: int, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ImprovementProposalError(f"{label} is empty or exceeds the host limit")


def _validate_content(value: str, label: str) -> None:
    if _SENSITIVE_PATTERN.search(value):
        raise ImprovementProposalError(f"{label} contains secret-like content")
    if _FORBIDDEN_CODE_PATTERN.search(value):
        raise ImprovementProposalError(f"{label} contains executable code")
    if _FORBIDDEN_PATH_PATTERN.search(value):
        raise ImprovementProposalError(f"{label} contains a path")
    if _FORBIDDEN_POLICY_PATTERN.search(value):
        raise ImprovementProposalError(f"{label} changes provider or capability policy")


__all__ = [
    "ImprovementProposal",
    "ImprovementProposalError",
    "ImprovementProposalNotFoundError",
    "ImprovementProposalService",
    "ImprovementRevisionConflictError",
    "ProposalStatus",
]
