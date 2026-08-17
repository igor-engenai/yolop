from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, cast

from pydantic_ai import ApprovalRequired
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)


@dataclass(frozen=True)
class ToolPolicyContext:
    """Bounded policy input for one validated tool call."""

    run_context: RunContext[Any]
    tool_name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolAuditRecord:
    """Safe tool lifecycle data without raw arguments or tool results."""

    phase: Literal["before_execute", "after_execute", "denied", "approval_requested"]
    tool_name: str
    outcome: str
    argument_count: int
    payload_digest: str


PolicyDecision = Callable[[ToolPolicyContext], str | None | Awaitable[str | None]]
PolicyRewrite = Callable[
    [ToolPolicyContext], Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None]
]
AuditSink = Callable[[ToolAuditRecord], None | Awaitable[None]]


@dataclass
class ToolPolicy(AbstractCapability[Any]):
    """Host-enforced deny, rewrite, approval, and safe audit policy."""

    deny: PolicyDecision | None = None
    rewrite: PolicyRewrite | None = None
    approve: PolicyDecision | None = None
    audit: AuditSink | None = None
    max_payload_bytes: int = 4096

    def __post_init__(self) -> None:
        if self.max_payload_bytes <= 0:
            raise ValueError("Tool policy payload limit must be positive")

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        policy_context = ToolPolicyContext(ctx, tool_def.name, dict(args))
        rewritten = await _call_rewrite(self.rewrite, policy_context)
        if rewritten is not None:
            rewritten_args = dict(rewritten)
            _validate_rewrite(rewritten_args, args, self.max_payload_bytes)
            args = rewritten_args
            policy_context = ToolPolicyContext(ctx, tool_def.name, dict(args))

        denial = await _call_decision(self.deny, policy_context)
        if denial is not None:
            reason = _safe_reason(denial, fallback="policy_denied")
            await self._audit(
                phase="denied",
                tool_name=tool_def.name,
                outcome=reason,
                args=args,
            )
            raise ApprovalRequired(
                metadata={"yolop_policy": {"decision": "deny", "reason": reason}}
            )

        if not ctx.tool_call_approved:
            approval = await _call_decision(self.approve, policy_context)
            if approval is not None:
                reason = _safe_reason(approval, fallback="approval_required")
                await self._audit(
                    phase="approval_requested",
                    tool_name=tool_def.name,
                    outcome=reason,
                    args=args,
                )
                raise ApprovalRequired(
                    metadata={"yolop_policy": {"decision": "approve", "reason": reason}}
                )

        await self._audit(
            phase="before_execute",
            tool_name=tool_def.name,
            outcome="allowed",
            args=args,
        )
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        await self._audit(
            phase="after_execute",
            tool_name=tool_def.name,
            outcome="completed",
            args=args,
        )
        return result

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[Any],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        approvals: dict[str, bool | ToolApproved | ToolDenied] = {}
        for call in requests.approvals:
            metadata = requests.metadata.get(call.tool_call_id, {})
            policy = metadata.get("yolop_policy")
            if isinstance(policy, dict) and policy.get("decision") == "deny":
                reason = _safe_reason(policy.get("reason"), fallback="policy_denied")
                approvals[call.tool_call_id] = ToolDenied(message=reason)
        if not approvals:
            return None
        return DeferredToolResults(approvals=approvals)

    async def _audit(
        self,
        *,
        phase: Literal["before_execute", "after_execute", "denied", "approval_requested"],
        tool_name: str,
        outcome: str,
        args: Mapping[str, Any],
    ) -> None:
        if self.audit is None:
            return
        record = ToolAuditRecord(
            phase=phase,
            tool_name=tool_name,
            outcome=outcome,
            argument_count=len(args),
            payload_digest=_digest(args, max_bytes=self.max_payload_bytes),
        )
        result = self.audit(record)
        if inspect.isawaitable(result):
            await result


async def _call_decision(
    decision: PolicyDecision | None,
    context: ToolPolicyContext,
) -> str | None:
    if decision is None:
        return None
    result = decision(context)
    if inspect.isawaitable(result):
        return cast(str | None, await result)
    return result


async def _call_rewrite(
    rewrite: PolicyRewrite | None,
    context: ToolPolicyContext,
) -> Mapping[str, Any] | None:
    if rewrite is None:
        return None
    result = rewrite(context)
    if inspect.isawaitable(result):
        return cast(Mapping[str, Any] | None, await result)
    return result


def _validate_rewrite(
    rewritten: Mapping[str, Any], original: Mapping[str, Any], max_bytes: int
) -> None:
    if not set(rewritten).issubset(original):
        raise ValueError("Tool policy rewrite cannot add argument names")
    try:
        encoded = json.dumps(rewritten, sort_keys=True, default=repr).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Tool policy rewrite arguments are not serializable") from error
    if len(encoded) > max_bytes:
        raise ValueError("Tool policy rewrite exceeds the payload limit")


def _digest(args: Mapping[str, Any], *, max_bytes: int) -> str:
    encoded = json.dumps(dict(args), sort_keys=True, default=repr).encode()[:max_bytes]
    return sha256(encoded).hexdigest()


def _safe_reason(value: object, *, fallback: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value):
        return fallback
    return value
