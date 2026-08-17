"""Host-authorized MCP toolsets for YoloP."""

from .config import (
    MCPAgentSelection,
    MCPConfigurationError,
    MCPServerConfig,
    MCPServerSelection,
    selections_from_spec,
)
from .registry import (
    MCPComposition,
    MCPConnection,
    MCPFeatureForbiddenError,
    MCPForbiddenAliasError,
    MCPRegistry,
    MCPSecretResolutionError,
    MCPToolPrefixConflictError,
    MCPUnknownAliasError,
)

__all__ = [
    "MCPAgentSelection",
    "MCPConfigurationError",
    "MCPServerConfig",
    "MCPServerSelection",
    "MCPComposition",
    "MCPConnection",
    "MCPFeatureForbiddenError",
    "MCPForbiddenAliasError",
    "MCPRegistry",
    "MCPSecretResolutionError",
    "MCPToolPrefixConflictError",
    "MCPUnknownAliasError",
    "selections_from_spec",
]
