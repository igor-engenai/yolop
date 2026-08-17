from typing import Any

from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability

from .catalog import ProviderCatalog


def load_capability_types(
    spec: AgentSpec | dict[str, Any],
    *,
    catalog: ProviderCatalog,
) -> tuple[type[AbstractCapability[Any]], ...]:
    """Return custom capability classes selected by an immutable provider catalog."""
    return catalog.capability_types_for(spec)
