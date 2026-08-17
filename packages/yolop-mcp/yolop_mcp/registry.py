from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, cast

from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, Toolset
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

from .config import MCPConfigurationError, MCPServerConfig, MCPServerSelection, selections_from_spec


class MCPUnknownAliasError(MCPConfigurationError):
    """AgentSpec selected an alias absent from the host registry."""

    code = "mcp_unknown_alias"


class MCPForbiddenAliasError(MCPConfigurationError):
    """AgentSpec selected a configured alias not allowed by this host."""

    code = "mcp_alias_forbidden"


class MCPSecretResolutionError(MCPConfigurationError):
    """A host secret reference could not be resolved."""

    code = "mcp_secret_resolution_failed"


class MCPToolPrefixConflictError(MCPConfigurationError):
    """Selected MCP aliases would use the same tool prefix."""

    code = "mcp_tool_prefix_conflict"


class MCPFeatureForbiddenError(MCPConfigurationError):
    """The host did not authorize an MCP resource or skill operation."""

    code = "mcp_feature_forbidden"


SecretResolver = Callable[[str], str | Awaitable[str]]


class _SecretResolver(Protocol):
    def __call__(self, reference: str, /) -> str | Awaitable[str]: ...


@dataclass(frozen=True)
class MCPConnection:
    """One ephemeral native MCP connection and its effective feature gates."""

    alias: str
    toolset: MCPToolset[Any]
    resources_enabled: bool
    skills_enabled: bool

    async def list_resources(self) -> list[Any]:
        if not self.resources_enabled:
            raise MCPFeatureForbiddenError("MCP resources are not enabled for this alias")
        return await self.toolset.list_resources()

    async def list_skills(self) -> list[Any]:
        if not self.skills_enabled:
            raise MCPFeatureForbiddenError("MCP skills are not enabled for this alias")
        return await self.toolset.list_prompts()

    async def get_skill(self, name: str, arguments: dict[str, str] | None = None) -> Any:
        if not self.skills_enabled:
            raise MCPFeatureForbiddenError("MCP skills are not enabled for this alias")
        return await self.toolset.get_prompt(name, arguments)

    async def aclose(self) -> None:
        """Close a connection left open by a host operation."""
        if self.toolset.is_running:
            await self.toolset.__aexit__(None, None, None)


@dataclass(frozen=True)
class MCPComposition:
    """Native capabilities and ephemeral connections for one AgentSpec."""

    capabilities: tuple[AbstractCapability[Any], ...]
    connections: tuple[MCPConnection, ...]

    async def aclose(self) -> None:
        for connection in self.connections:
            await connection.aclose()


class MCPRegistry:
    """Resolve AgentSpec MCP aliases against one host-owned authorization registry."""

    def __init__(
        self,
        servers: Iterable[MCPServerConfig],
        *,
        allowed_aliases: Iterable[str] | None = None,
    ) -> None:
        by_alias: dict[str, MCPServerConfig] = {}
        for server in servers:
            if server.alias in by_alias:
                raise MCPConfigurationError(f"Duplicate MCP server alias: {server.alias}")
            by_alias[server.alias] = server
        allowed = None if allowed_aliases is None else frozenset(allowed_aliases)
        unknown_allowed = () if allowed is None else sorted(allowed - by_alias.keys())
        if unknown_allowed:
            raise MCPConfigurationError("MCP authorization references an unknown alias")
        self._servers = MappingProxyType(by_alias)
        self._allowed_aliases = allowed

    @property
    def servers(self) -> Mapping[str, MCPServerConfig]:
        return self._servers

    async def build_for_spec(
        self,
        spec: AgentSpec | Mapping[str, Any],
        *,
        secret_resolver: SecretResolver | None = None,
    ) -> MCPComposition:
        """Build native mandatory capabilities for the safe aliases in an AgentSpec."""
        selections = selections_from_spec(spec)
        prefixes = [
            self._servers[selection.alias].tool_prefix or selection.alias
            for selection in selections
            if selection.alias in self._servers
        ]
        if len(set(prefixes)) != len(prefixes):
            raise MCPToolPrefixConflictError("Selected MCP aliases must have unique tool prefixes")

        capabilities: list[AbstractCapability[Any]] = []
        connections: list[MCPConnection] = []
        for selection in selections:
            config = self._config_for(selection.alias)
            self._authorize(selection.alias)
            effective_tools = _effective_tools(config, selection)
            headers = await _resolve_values(
                config.headers,
                config.header_secret_refs,
                secret_resolver,
                kind="header",
            )
            environment = await _resolve_values(
                config.environment,
                config.environment_secret_refs,
                secret_resolver,
                kind="environment",
            )
            native_toolset = _build_toolset(config, headers=headers, environment=environment)
            toolset: AbstractToolset[Any] = native_toolset
            if effective_tools is not None:
                toolset = toolset.filtered(
                    lambda _ctx, tool_definition: tool_definition.name in effective_tools
                )
            prefix = config.tool_prefix or config.alias
            toolset = toolset.prefixed(prefix)
            capabilities.append(Toolset(toolset=toolset))
            connections.append(
                MCPConnection(
                    alias=config.alias,
                    toolset=native_toolset,
                    resources_enabled=config.allow_resources and selection.allow_resources,
                    skills_enabled=config.allow_skills and selection.allow_skills,
                )
            )
        return MCPComposition(tuple(capabilities), tuple(connections))

    def _config_for(self, alias: str) -> MCPServerConfig:
        config = self._servers.get(alias)
        if config is None:
            raise MCPUnknownAliasError(f"Unknown MCP server alias: {alias}")
        return config

    def _authorize(self, alias: str) -> None:
        if self._allowed_aliases is not None and alias not in self._allowed_aliases:
            raise MCPForbiddenAliasError("MCP server alias is not authorized by this host")


def _effective_tools(
    config: MCPServerConfig, selection: MCPServerSelection
) -> frozenset[str] | None:
    host_allowed = None if config.allowed_tools is None else frozenset(config.allowed_tools)
    selected = None if selection.allowed_tools is None else frozenset(selection.allowed_tools)
    if host_allowed is None:
        return selected
    if selected is not None and not selected.issubset(host_allowed):
        raise MCPConfigurationError("AgentSpec selected an MCP tool outside the host allowlist")
    return host_allowed if selected is None else selected


def _build_toolset(
    config: MCPServerConfig,
    *,
    headers: dict[str, str],
    environment: dict[str, str],
) -> MCPToolset[Any]:
    if config.transport == "stdio":
        from fastmcp.client.transports import StdioTransport

        transport = StdioTransport(
            command=cast(str, config.command),
            args=list(config.args),
            env=environment,
        )
        return MCPToolset(
            transport,
            id=config.alias,
            init_timeout=config.timeout_seconds,
            read_timeout=config.timeout_seconds,
            cache_resources=config.allow_resources,
            cache_prompts=config.allow_skills,
            include_instructions=config.allow_skills,
        )
    return MCPToolset(
        cast(str, config.url),
        id=config.alias,
        headers=headers or None,
        init_timeout=config.timeout_seconds,
        read_timeout=config.timeout_seconds,
        cache_resources=config.allow_resources,
        cache_prompts=config.allow_skills,
        include_instructions=config.allow_skills,
    )


async def _resolve_values(
    values: Mapping[str, str],
    secret_refs: Mapping[str, str],
    resolver: _SecretResolver | None,
    *,
    kind: str,
) -> dict[str, str]:
    resolved = dict(values)
    if secret_refs and resolver is None:
        raise MCPSecretResolutionError(f"MCP {kind} secret resolver is required")
    for name, reference in secret_refs.items():
        assert resolver is not None
        try:
            value = resolver(reference)
            if inspect.isawaitable(value):
                value = await value
        except Exception as error:
            raise MCPSecretResolutionError(f"MCP {kind} secret resolution failed") from error
        if not isinstance(value, str) or not value:
            raise MCPSecretResolutionError(
                f"MCP {kind} secret resolution returned an invalid value"
            )
        resolved[name] = value
    return resolved
