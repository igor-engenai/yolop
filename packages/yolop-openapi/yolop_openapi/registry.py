from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pydantic_ai import AgentSpec, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze(item) for item in value)
    return value


class OpenAPIConfigurationError(ValueError):
    """The host OpenAPI registry or AgentSpec selection is invalid."""


class OpenAPIUnknownAliasError(OpenAPIConfigurationError):
    """The AgentSpec selected an unknown OpenAPI alias."""


class OpenAPIForbiddenAliasError(OpenAPIConfigurationError):
    """The host did not authorize an OpenAPI alias."""


class OpenAPIServerError(OpenAPIConfigurationError):
    """The configured server is not authorized by the pinned document."""


class OpenAPIOperationError(OpenAPIConfigurationError):
    """An OpenAPI operation is invalid or outside the host allowlist."""


@dataclass(frozen=True)
class OpenAPIServiceConfig:
    """Immutable host-owned OpenAPI document and server binding."""

    alias: str
    document: Mapping[str, Any]
    server_url: str
    allowed_operations: frozenset[str] | None = None
    tool_prefix: str | None = None
    spec_digest: str | None = None
    security_schemes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        document = dict(self.document)
        if not self.alias.strip():
            raise OpenAPIConfigurationError("OpenAPI alias must not be empty")
        if not str(document.get("openapi", "")).startswith("3."):
            raise OpenAPIConfigurationError("Only OpenAPI 3 documents are supported")
        servers = document.get("servers")
        if not isinstance(servers, Sequence) or isinstance(servers, (str, bytes)) or not any(
            isinstance(server, dict) and server.get("url") == self.server_url for server in servers
        ):
            raise OpenAPIServerError("Configured server is not present in the pinned document")
        operations = _operations(document)
        if not operations:
            raise OpenAPIOperationError("OpenAPI document has no operations")
        if self.allowed_operations is not None and not self.allowed_operations.issubset(
            operations.keys()
        ):
            raise OpenAPIOperationError("Host operation allowlist contains an unknown operation")
        digest = _digest(document)
        if self.spec_digest is not None and self.spec_digest != digest:
            raise OpenAPIConfigurationError("OpenAPI document digest does not match")
        object.__setattr__(self, "document", _deep_freeze(document))
        object.__setattr__(self, "security_schemes", _deep_freeze(self.security_schemes))
        object.__setattr__(self, "spec_digest", digest)

    @property
    def operations(self) -> Mapping[str, Mapping[str, str]]:
        return MappingProxyType(_operations(self.document))


@dataclass(frozen=True)
class OpenAPISelection:
    alias: str
    allowed_operations: frozenset[str] | None = None


@dataclass(frozen=True)
class OpenAPIComposition:
    capabilities: tuple[AbstractCapability[Any], ...]


@dataclass
class OpenAPIExplorer(AbstractCapability[Any]):
    alias: str
    catalog: tuple[Mapping[str, str], ...]

    def get_toolset(self) -> AbstractToolset[Any]:
        return FunctionToolset(
            [Tool(self.explore, takes_ctx=False)],
            id=f"{self.alias}__explore",
        ).prefixed(self.alias)

    async def explore(self, query: str | None = None) -> list[dict[str, str]]:
        """Explore only host-authorized operation metadata."""
        normalized = query.casefold().strip() if query else None
        values = [
            dict(operation)
            for operation in self.catalog
            if normalized is None
            or normalized in operation["operation_id"].casefold()
            or normalized in operation["path"].casefold()
            or normalized in operation["method"].casefold()
        ]
        return sorted(values, key=lambda operation: operation["operation_id"])


class OpenAPIRegistry:
    """Resolve AgentSpec OpenAPI aliases against host-owned immutable configs."""

    def __init__(
        self,
        services: Iterable[OpenAPIServiceConfig],
        *,
        allowed_aliases: Iterable[str] | None = None,
    ) -> None:
        by_alias: dict[str, OpenAPIServiceConfig] = {}
        prefixes: set[str] = set()
        for service in services:
            if service.alias in by_alias:
                raise OpenAPIConfigurationError("Duplicate OpenAPI alias")
            prefix = service.tool_prefix or service.alias
            if prefix in prefixes:
                raise OpenAPIConfigurationError("Duplicate OpenAPI tool prefix")
            prefixes.add(prefix)
            by_alias[service.alias] = service
        allowed = None if allowed_aliases is None else frozenset(allowed_aliases)
        if allowed is not None and not allowed.issubset(by_alias):
            raise OpenAPIConfigurationError("OpenAPI authorization references an unknown alias")
        self._services = MappingProxyType(by_alias)
        self._allowed_aliases = allowed

    @property
    def services(self) -> Mapping[str, OpenAPIServiceConfig]:
        return self._services

    def build_for_spec(self, spec: AgentSpec | Mapping[str, Any]) -> OpenAPIComposition:
        selections = _selections_from_spec(spec)
        capabilities: list[AbstractCapability[Any]] = []
        for selection in selections:
            service = self._services.get(selection.alias)
            if service is None:
                raise OpenAPIUnknownAliasError(f"Unknown OpenAPI alias: {selection.alias}")
            if self._allowed_aliases is not None and selection.alias not in self._allowed_aliases:
                raise OpenAPIForbiddenAliasError("OpenAPI alias is not authorized by this host")
            host_operations = service.allowed_operations or frozenset(service.operations)
            effective = host_operations
            if selection.allowed_operations is not None:
                if not selection.allowed_operations.issubset(host_operations):
                    raise OpenAPIOperationError(
                        "AgentSpec selected an operation outside the host allowlist"
                    )
                effective = selection.allowed_operations
            catalog = tuple(
                {
                    "operation_id": operation_id,
                    "method": service.operations[operation_id]["method"],
                    "path": service.operations[operation_id]["path"],
                }
                for operation_id in sorted(effective)
            )
            capabilities.append(
                OpenAPIExplorer(
                    alias=service.tool_prefix or service.alias,
                    catalog=catalog,
                )
            )
        return OpenAPIComposition(tuple(capabilities))


def _selections_from_spec(spec: AgentSpec | Mapping[str, Any]) -> tuple[OpenAPISelection, ...]:
    data = spec.model_dump(mode="python") if isinstance(spec, AgentSpec) else dict(spec)
    metadata = data.get("metadata") or {}
    raw = metadata.get("openapi", ()) if isinstance(metadata, Mapping) else ()
    selections: list[OpenAPISelection] = []
    for item in raw:
        if isinstance(item, str):
            selections.append(OpenAPISelection(alias=item))
        elif isinstance(item, Mapping) and isinstance(item.get("alias"), str):
            operations = item.get("operations")
            selections.append(
                OpenAPISelection(
                    alias=item["alias"],
                    allowed_operations=(
                        frozenset(operations) if operations is not None else None
                    ),
                )
            )
        else:
            raise OpenAPIConfigurationError("OpenAPI selection is invalid")
    return tuple(selections)


def _operations(document: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    paths = document.get("paths")
    if not isinstance(paths, Mapping):
        raise OpenAPIOperationError("OpenAPI paths are required")
    operations: dict[str, dict[str, str]] = {}
    methods = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
    for path, path_item in paths.items():
        if not isinstance(path, str) or not isinstance(path_item, Mapping):
            continue
        for method, operation in path_item.items():
            if method not in methods:
                continue
            if not isinstance(operation, Mapping) or not isinstance(
                operation.get("operationId"), str
            ):
                raise OpenAPIOperationError("Every operation must have an operationId")
            operation_id = operation["operationId"]
            if operation_id in operations:
                raise OpenAPIOperationError("OpenAPI operation IDs must be unique")
            operations[operation_id] = {
                "operation_id": operation_id,
                "method": method.upper(),
                "path": path,
            }
    return operations


def _digest(document: Mapping[str, Any]) -> str:
    canonical = json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "OpenAPIComposition",
    "OpenAPIConfigurationError",
    "OpenAPIExplorer",
    "OpenAPIForbiddenAliasError",
    "OpenAPIOperationError",
    "OpenAPIRegistry",
    "OpenAPISelection",
    "OpenAPIServerError",
    "OpenAPIServiceConfig",
    "OpenAPIUnknownAliasError",
]
