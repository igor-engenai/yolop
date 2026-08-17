from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import AbstractCapability

from .catalog import ProviderCatalog


class CapabilityPolicyConflictError(ValueError):
    """A host-enforced capability conflicts with AgentSpec policy."""

    code = "capability_policy_conflict"


@dataclass(frozen=True)
class CapabilityResolution:
    """The separate AgentSpec and host-enforced capability selections."""

    selected: tuple[str, ...]
    enforced: tuple[str, ...]
    selected_types: tuple[type[AbstractCapability[Any]], ...]
    enforced_capabilities: tuple[AbstractCapability[Any], ...]


def resolve_capabilities(
    spec: AgentSpec | dict[str, Any],
    *,
    catalog: ProviderCatalog,
    mandatory_capabilities: Sequence[AbstractCapability[Any]] = (),
) -> CapabilityResolution:
    """Resolve AgentSpec capabilities and append immutable host policy capabilities."""
    catalog.validate_spec(spec)
    selected = catalog.selected_capability_names(spec)
    enforced_capabilities = tuple(mandatory_capabilities)
    enforced = tuple(_capability_identity(capability) for capability in enforced_capabilities)
    _ensure_unique_enforced(enforced)
    conflicts = tuple(sorted(set(selected) & set(enforced)))
    if conflicts:
        names = ", ".join(repr(name) for name in conflicts)
        raise CapabilityPolicyConflictError(
            f"Mandatory host capability conflicts with AgentSpec capability: {names}"
        )
    selected_names = tuple(dict.fromkeys(selected))
    selected_types = tuple(
        catalog.capability_type(name) for name in selected_names if catalog.has_capability(name)
    )
    return CapabilityResolution(
        selected=selected_names,
        enforced=enforced,
        selected_types=selected_types,
        enforced_capabilities=enforced_capabilities,
    )


def load_capability_types(
    spec: AgentSpec | dict[str, Any],
    *,
    catalog: ProviderCatalog,
) -> tuple[type[AbstractCapability[Any]], ...]:
    """Return custom capability classes selected by an immutable provider catalog."""
    return resolve_capabilities(spec, catalog=catalog).selected_types


def _capability_identity(capability: AbstractCapability[Any]) -> str:
    if capability.id:
        return capability.id
    serialization_name = capability.get_serialization_name()
    if serialization_name:
        return serialization_name
    return type(capability).__name__


def _ensure_unique_enforced(names: Sequence[str]) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        joined = ", ".join(repr(name) for name in duplicates)
        raise CapabilityPolicyConflictError(f"Duplicate mandatory host capabilities: {joined}")
