from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ALIAS_PATTERN = r"[A-Za-z][A-Za-z0-9-]{0,63}"
_ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,127}")
_SENSITIVE_ENVIRONMENT_PATTERN = re.compile(r"(?:API_KEY|TOKEN|SECRET|PASSWORD)$")
_SENSITIVE_HEADER_NAMES = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key", "x-auth-token"}
)

# These names are common model-provider credentials. A stdio MCP server receives an explicit
# environment, so it never inherits these credentials from the host process.
_MODEL_CREDENTIAL_NAMES = frozenset(
    {
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }
)


class MCPConfigurationError(ValueError):
    """Host MCP configuration is invalid or unsafe."""

    code = "mcp_configuration_error"


class MCPServerConfig(BaseModel):
    """Host-owned connection configuration without resolved secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(pattern=_ALIAS_PATTERN)
    transport: Literal["stdio", "http"]
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=300.0)
    tool_prefix: str | None = Field(default=None, pattern=_ALIAS_PATTERN)
    allowed_tools: tuple[str, ...] | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    header_secret_refs: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    environment_secret_refs: dict[str, str] = Field(default_factory=dict)
    allow_resources: bool = False
    allow_skills: bool = False

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        if self.transport == "stdio":
            if not self.command:
                raise MCPConfigurationError("stdio MCP server requires a command")
            if self.url is not None:
                raise MCPConfigurationError("stdio MCP server must not define a URL")
            if self.headers or self.header_secret_refs:
                raise MCPConfigurationError("stdio MCP server must not define HTTP headers")
        elif self.url is None or not self.url.startswith(("http://", "https://")):
            raise MCPConfigurationError("HTTP MCP server requires an http(s) URL")
        if self.transport == "http" and (self.command is not None or self.args):
            raise MCPConfigurationError("HTTP MCP server must not define a command or arguments")
        if self.transport == "http" and (self.environment or self.environment_secret_refs):
            raise MCPConfigurationError("HTTP MCP server must not define an environment")
        if any(name.lower() in _SENSITIVE_HEADER_NAMES for name in self.headers):
            raise MCPConfigurationError("Sensitive MCP headers must use a secret reference")
        _validate_environment_names(self.environment)
        _validate_environment_names(self.environment_secret_refs)
        blocked = _MODEL_CREDENTIAL_NAMES.intersection(
            set(self.environment) | set(self.environment_secret_refs)
        )
        if blocked:
            raise MCPConfigurationError("stdio MCP environment contains a model credential")
        if any(_SENSITIVE_ENVIRONMENT_PATTERN.search(name) for name in self.environment):
            raise MCPConfigurationError(
                "Sensitive MCP environment values must use secret references"
            )
        if set(self.headers).intersection(self.header_secret_refs):
            raise MCPConfigurationError("A header cannot have both a value and a secret reference")
        if not all(ref.strip() for ref in self.header_secret_refs.values()):
            raise MCPConfigurationError("MCP header secret references must not be empty")
        if not all(ref.strip() for ref in self.environment_secret_refs.values()):
            raise MCPConfigurationError("MCP environment secret references must not be empty")
        if self.allowed_tools is not None and len(set(self.allowed_tools)) != len(
            self.allowed_tools
        ):
            raise MCPConfigurationError("MCP allowed tool names must be unique")
        return self


class MCPServerSelection(BaseModel):
    """Safe AgentSpec selection for one host-authorized MCP alias."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(pattern=_ALIAS_PATTERN)
    allowed_tools: tuple[str, ...] | None = None
    allow_resources: bool = False
    allow_skills: bool = False

    @model_validator(mode="after")
    def validate_tools(self) -> MCPServerSelection:
        if self.allowed_tools is not None and len(set(self.allowed_tools)) != len(
            self.allowed_tools
        ):
            raise MCPConfigurationError("MCP selected tool names must be unique")
        return self


class MCPAgentSelection(BaseModel):
    """The `metadata.mcp` section of an AgentSpec."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    servers: tuple[MCPServerSelection, ...] = ()

    @model_validator(mode="after")
    def validate_aliases(self) -> MCPAgentSelection:
        aliases = [selection.alias for selection in self.servers]
        if len(set(aliases)) != len(aliases):
            raise MCPConfigurationError("MCP server aliases must be unique in AgentSpec metadata")
        return self


def selections_from_spec(spec: object) -> tuple[MCPServerSelection, ...]:
    """Read safe MCP alias selections from AgentSpec metadata."""
    if isinstance(spec, Mapping):
        raw_metadata = spec.get("metadata")
    else:
        raw_metadata = getattr(spec, "metadata", None)
    if raw_metadata is None:
        return ()
    if not isinstance(raw_metadata, Mapping):
        raise MCPConfigurationError("AgentSpec metadata must be an object")
    raw_mcp = raw_metadata.get("mcp")
    if raw_mcp is None:
        return ()
    try:
        selection = MCPAgentSelection.model_validate(raw_mcp)
    except MCPConfigurationError:
        raise
    except Exception as error:
        raise MCPConfigurationError("AgentSpec MCP metadata is invalid") from error
    return selection.servers


def _validate_environment_names(values: Mapping[str, str]) -> None:
    invalid = [name for name in values if _ENVIRONMENT_NAME_PATTERN.fullmatch(name) is None]
    if invalid:
        raise MCPConfigurationError("MCP environment names must be uppercase shell names")
