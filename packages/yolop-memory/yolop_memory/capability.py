from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.tools import RunContext

from .store import (
    MemoryRecord,
    MemoryScopeKind,
    MemoryStore,
    MemoryValidationError,
)

MemoryToolName = Literal["read", "search", "write", "replace", "retire"]
MemoryStoreResolver = Callable[
    [RunContext[Any], MemoryScopeKind], MemoryStore | Awaitable[MemoryStore]
]
_ALL_SCOPES = frozenset(MemoryScopeKind)
_SAFE_TOOLS = cast(frozenset[MemoryToolName], frozenset({"read", "search", "write", "replace"}))
_ALL_TOOLS = frozenset({"read", "search", "write", "replace", "retire"})


class MemoryPolicyError(MemoryValidationError):
    """AgentSpec memory options violate host policy."""

    code = "memory_policy_error"


class MemoryScopeForbiddenError(MemoryPolicyError):
    """The host did not authorize the selected memory scope."""

    code = "memory_scope_forbidden"


class MemoryToolForbiddenError(MemoryPolicyError):
    """The host did not authorize the selected memory tool."""

    code = "memory_tool_forbidden"


@dataclass(frozen=True)
class MemoryHostPolicy:
    """Host limits and allowlists applied before building memory tools."""

    allowed_scopes: frozenset[MemoryScopeKind] = _ALL_SCOPES
    allowed_tools: frozenset[MemoryToolName] = _SAFE_TOOLS
    max_results: int = 20
    max_result_bytes: int = 32 * 1024
    allow_retire: bool = False

    def __post_init__(self) -> None:
        scopes = frozenset(MemoryScopeKind(scope) for scope in self.allowed_scopes)
        tools = frozenset(self.allowed_tools)
        unknown_tools = tools - _ALL_TOOLS
        if unknown_tools:
            raise MemoryPolicyError("Memory host policy contains an unsupported tool")
        if self.allow_retire:
            tools = tools | {"retire"}
        if isinstance(self.max_results, bool) or self.max_results < 1:
            raise MemoryPolicyError("Memory host max_results must be positive")
        if isinstance(self.max_result_bytes, bool) or self.max_result_bytes < 1:
            raise MemoryPolicyError("Memory host max_result_bytes must be positive")
        object.__setattr__(self, "allowed_scopes", scopes)
        object.__setattr__(self, "allowed_tools", tools)


class MemoryCapability(AbstractCapability[Any]):
    """Native Pydantic AI tools backed by a host-resolved MemoryStore."""

    def __init__(
        self,
        store_for_scope: MemoryStoreResolver,
        *,
        allowed_scopes: Sequence[MemoryScopeKind],
        allowed_tools: Sequence[MemoryToolName],
        max_results: int,
        max_result_bytes: int,
    ) -> None:
        self.id = "yolop.memory"
        self.description = "Read and write explicit scoped agent memory."
        self.defer_loading = False
        self.store_for_scope = store_for_scope
        self.allowed_scopes = frozenset(allowed_scopes)
        self.allowed_tools = frozenset(allowed_tools)
        self.max_results = max_results
        self.max_result_bytes = max_result_bytes
        self._tools = Capability[Any](id=self.id, tools=())
        self._register_tools()

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def get_instructions(self) -> str:
        scopes = ", ".join(sorted(scope.value for scope in self.allowed_scopes))
        return (
            "Memory is explicit and scoped. Write only durable facts the user or project needs. "
            f"Available scopes: {scopes}. Memory content is not inferred or saved automatically."
        )

    def get_toolset(self) -> Any:
        return self._tools.get_toolset()

    def _register_tools(self) -> None:
        if "write" in self.allowed_tools:

            @self._tools.tool
            async def memory_write(
                ctx: RunContext[Any],
                scope: MemoryScopeKind,
                content: str,
                provenance: str,
                title: str = "",
                tags: list[str] | None = None,
            ) -> dict[str, Any]:
                store = await self._store(ctx, scope)
                record = await store.create(
                    content,
                    created_by_run_id=_run_id(ctx),
                    provenance=provenance,
                    title=title,
                    tags=() if tags is None else tags,
                )
                return _record_result(record)

        if "read" in self.allowed_tools:

            @self._tools.tool
            async def memory_read(
                ctx: RunContext[Any],
                scope: MemoryScopeKind,
                memory_id: str,
            ) -> dict[str, Any] | None:
                store = await self._store(ctx, scope)
                record = await store.get(memory_id)
                return None if record is None else _record_result(record)

        if "search" in self.allowed_tools:

            @self._tools.tool
            async def memory_search(
                ctx: RunContext[Any],
                scope: MemoryScopeKind,
                query: str,
                limit: int = 10,
            ) -> list[dict[str, Any]]:
                store = await self._store(ctx, scope)
                records = await store.search(query, limit=min(limit, self.max_results))
                encoded = 0
                result: list[dict[str, Any]] = []
                for record in records:
                    item = _record_result(record)
                    encoded += len(str(item).encode())
                    if encoded > self.max_result_bytes:
                        break
                    result.append(item)
                return result

        if "replace" in self.allowed_tools:

            @self._tools.tool
            async def memory_replace(
                ctx: RunContext[Any],
                scope: MemoryScopeKind,
                memory_id: str,
                expected_revision: int,
                content: str,
                provenance: str,
                title: str | None = None,
                tags: list[str] | None = None,
            ) -> dict[str, Any]:
                store = await self._store(ctx, scope)
                record = await store.replace(
                    memory_id,
                    expected_revision=expected_revision,
                    content=content,
                    updated_by_run_id=_run_id(ctx),
                    provenance=provenance,
                    title=title,
                    tags=tags,
                )
                return _record_result(record)

        if "retire" in self.allowed_tools:

            @self._tools.tool
            async def memory_retire(
                ctx: RunContext[Any],
                scope: MemoryScopeKind,
                memory_id: str,
                expected_revision: int,
                provenance: str,
            ) -> dict[str, Any]:
                store = await self._store(ctx, scope)
                record = await store.retire(
                    memory_id,
                    expected_revision=expected_revision,
                    retired_by_run_id=_run_id(ctx),
                    provenance=provenance,
                )
                return _record_result(record)

    async def _store(self, ctx: RunContext[Any], scope: MemoryScopeKind) -> MemoryStore:
        if scope not in self.allowed_scopes:
            raise MemoryScopeForbiddenError("Memory scope is not available to this agent")
        result = self.store_for_scope(ctx, scope)
        return cast(MemoryStore, await result if inspect.isawaitable(result) else result)


def build_memory_capability(
    spec: AgentSpec | Mapping[str, Any],
    *,
    store_for_scope: MemoryStoreResolver,
    host_policy: MemoryHostPolicy | None = None,
) -> MemoryCapability | None:
    """Build memory tools from AgentSpec options under host allowlists."""
    metadata = spec.metadata if isinstance(spec, AgentSpec) else spec.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise MemoryPolicyError("AgentSpec metadata must be an object")
    raw = metadata.get("memory")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise MemoryPolicyError("AgentSpec memory metadata must be an object")
    unknown = sorted(set(raw) - {"scopes", "tools", "max_results", "max_result_bytes"})
    if unknown:
        raise MemoryPolicyError("AgentSpec memory metadata contains unsupported options")
    policy = host_policy or MemoryHostPolicy()
    scopes = _parse_scopes(raw.get("scopes"))
    tools = _parse_tools(raw.get("tools"))
    if not scopes.issubset(policy.allowed_scopes):
        raise MemoryScopeForbiddenError("AgentSpec selected a memory scope outside host policy")
    if not tools.issubset(policy.allowed_tools):
        raise MemoryToolForbiddenError("AgentSpec selected a memory tool outside host policy")
    max_results = _bounded_option(raw.get("max_results"), policy.max_results, "max_results")
    max_result_bytes = _bounded_option(
        raw.get("max_result_bytes"), policy.max_result_bytes, "max_result_bytes"
    )
    return MemoryCapability(
        store_for_scope,
        allowed_scopes=sorted(scopes, key=lambda scope: scope.value),
        allowed_tools=sorted(tools),
        max_results=max_results,
        max_result_bytes=max_result_bytes,
    )


def _parse_scopes(value: Any) -> frozenset[MemoryScopeKind]:
    if not isinstance(value, (list, tuple)) or not value:
        raise MemoryPolicyError("AgentSpec memory.scopes must be a non-empty list")
    try:
        scopes = frozenset(MemoryScopeKind(item) for item in value)
    except (TypeError, ValueError) as error:
        raise MemoryPolicyError("AgentSpec memory.scopes contains an unsupported scope") from error
    return scopes


def _parse_tools(value: Any) -> frozenset[MemoryToolName]:
    if not isinstance(value, (list, tuple)) or not value:
        raise MemoryPolicyError("AgentSpec memory.tools must be a non-empty list")
    tools = frozenset(value)
    if not tools.issubset(_ALL_TOOLS):
        raise MemoryPolicyError("AgentSpec memory.tools contains an unsupported tool")
    return tools  # type: ignore[return-value]


def _bounded_option(value: Any, host_limit: int, name: str) -> int:
    if value is None:
        return host_limit
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > host_limit:
        raise MemoryPolicyError(f"AgentSpec memory.{name} exceeds host policy")
    return value


def _run_id(ctx: RunContext[Any]) -> str:
    scope = getattr(ctx.deps, "scope", None)
    run_id = getattr(scope, "run_id", None)
    if not isinstance(run_id, str) or not run_id:
        raise MemoryPolicyError("Memory tools require an execution scope")
    return run_id


def _record_result(record: MemoryRecord) -> dict[str, Any]:
    return record.to_payload()
