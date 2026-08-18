"""Durable deep-agent capabilities for YoloP."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.resources import as_file, files
from typing import Any, Literal

from pydantic_ai import AgentSpec
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.planning import (
    PlanItem,
    PlanStore,
    TaskStatus,
)
from pydantic_ai_harness.planning import (
    Planning as HarnessPlanning,
)
from yolop_runtime import (
    RuntimeDeps,
    ScopedState,
    ScopedStateContext,
)

from yolop import ProviderCatalog

from .goals import GoalRecord, GoalRunner, GoalStatus, GoalStore, GoalVerdict

_PLAN_OWNER = "yolop.deep.planning"
_PLAN_STATE_KIND = "plan"
_PLAN_SCHEMA_VERSION = 1


class SessionPlanStore(PlanStore):
    """A Harness PlanStore backed by one Runtime Session's plugin state."""

    def __init__(
        self,
        state: ScopedStateContext,
        *,
        owner_id: str = _PLAN_OWNER,
    ) -> None:
        self._state: ScopedState = state.for_session(owner_id)

    async def get_items(self) -> list[PlanItem]:
        entries = await self._state.read(_PLAN_STATE_KIND, schema_version=_PLAN_SCHEMA_VERSION)
        if not entries:
            return []
        payload = entries[-1].payload
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("Stored planning state has an invalid snapshot")
        try:
            return [PlanItem.model_validate(item) for item in payload["items"]]
        except (TypeError, ValueError) as error:
            raise ValueError("Stored planning state has an invalid item") from error

    async def set_items(self, items: list[PlanItem]) -> None:
        current = await self._state.read(_PLAN_STATE_KIND, schema_version=_PLAN_SCHEMA_VERSION)
        expected_sequence = current[-1].sequence if current else 0
        payload = {
            "items": [item.model_dump(mode="json") for item in items],
        }
        await self._state.append(
            _PLAN_STATE_KIND,
            payload,
            schema_version=_PLAN_SCHEMA_VERSION,
            expected_sequence=expected_sequence,
        )

    async def get_item(self, item_id: str) -> PlanItem | None:
        return next((item for item in await self.get_items() if item.id == item_id), None)

    async def add_item(self, item: PlanItem) -> PlanItem:
        items = await self.get_items()
        if any(existing.id == item.id for existing in items):
            raise ValueError(f"A step with id {item.id!r} is already in this plan.")
        stored = item.model_copy(deep=True)
        await self.set_items([*items, stored])
        return stored.model_copy(deep=True)

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanItem | None:
        items = await self.get_items()
        current = next((item for item in items if item.id == item_id), None)
        if current is None:
            return None
        changes: dict[str, Any] = {}
        for name, value in (
            ("content", content),
            ("status", status),
            ("active_form", active_form),
            ("parent_id", parent_id),
            ("depends_on", depends_on),
        ):
            if value is not None:
                changes[name] = list(value) if name == "depends_on" else value
        updated = current.model_copy(update=changes, deep=True)
        await self.set_items([updated if item.id == item_id else item for item in items])
        return updated.model_copy(deep=True)

    async def remove_item(self, item_id: str) -> bool:
        items = await self.get_items()
        remaining = [item for item in items if item.id != item_id]
        if len(remaining) == len(items):
            return False
        await self.set_items(remaining)
        return True


@dataclass
class Planning(HarnessPlanning[Any]):
    """Harness Planning bound to the current YoloP Session."""

    async def for_run(self, ctx: RunContext[Any]) -> Planning:
        """Bind a fresh Harness capability to the current Runtime Session."""
        if not isinstance(ctx.deps, RuntimeDeps):
            raise TypeError("yolop-deep Planning requires RuntimeDeps")
        clone = replace(self)
        clone.store = SessionPlanStore(ctx.deps.state)
        clone.store_resolver = None
        clone._resolved_store = clone.store
        return clone

    @classmethod
    def from_spec(
        cls,
        *,
        backend: Literal["memory", "sqlite"] = "memory",
        database: str = ".agent-plan.db",
        session: str = "default",
        enable_subtasks: bool = False,
        inject: bool = True,
        guidance: str | None = None,
        cache_ttl: Literal["5m", "1h"] = "5m",
        tools: list[str] | None = None,
        descriptions: dict[str, str] | None = None,
    ) -> Planning:
        """Construct planning policy and reject caller-selected storage."""
        if backend != "memory" or database != ".agent-plan.db" or session != "default":
            raise ValueError("yolop-deep Planning does not accept database or path configuration")
        return cls(
            enable_subtasks=enable_subtasks,
            inject=inject,
            guidance=guidance,
            cache_ttl=cache_ttl,
            tools=tools,
            descriptions=descriptions,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the AgentSpec capability name."""
        return "Planning"


from .teams import (  # noqa: E402
    AssignmentConflictError,
    ChannelAuthorizationError,
    ChannelMessage,
    DelegatedTaskCoordinator,
    PlanDependencyError,
    TaskAssignment,
    deep_delegation_aliases,
)


def load_deep_coding_spec(*, catalog: ProviderCatalog | None = None) -> AgentSpec:
    """Load the explicit deep-coding preset after validating installed capabilities."""
    with as_file(files("yolop_deep").joinpath("agent_specs/deep-coding.yaml")) as path:
        spec = AgentSpec.from_file(path)
    (catalog or ProviderCatalog.from_installed()).capability_types_for(spec)
    return spec


__all__ = [
    "GoalRecord",
    "GoalRunner",
    "GoalStatus",
    "GoalStore",
    "GoalVerdict",
    "AssignmentConflictError",
    "ChannelAuthorizationError",
    "ChannelMessage",
    "DelegatedTaskCoordinator",
    "PlanDependencyError",
    "PlanItem",
    "PlanStore",
    "Planning",
    "SessionPlanStore",
    "TaskAssignment",
    "TaskStatus",
    "deep_delegation_aliases",
    "load_deep_coding_spec",
]
