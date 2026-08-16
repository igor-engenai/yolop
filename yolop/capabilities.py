from importlib import metadata
from typing import Any

from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import CAPABILITY_TYPES, AbstractCapability

_ENTRY_POINT_GROUP = "yolop.capabilities"


def load_capability_types(
    spec: AgentSpec | dict[str, Any],
) -> tuple[type[AbstractCapability[Any]], ...]:
    """Load installed custom capability classes selected by an AgentSpec."""
    validated_spec = spec if isinstance(spec, AgentSpec) else AgentSpec.model_validate(spec)
    if not validated_spec.capabilities:
        return ()

    entry_points = metadata.entry_points(group=_ENTRY_POINT_GROUP)
    available_names = {entry_point.name for entry_point in entry_points}
    required_names = {
        capability.name
        for capability in validated_spec.capabilities
        if capability.name not in CAPABILITY_TYPES
    }
    for capability in validated_spec.capabilities:
        required_names.update(_find_names(capability.arguments, available_names))

    if not required_names:
        return ()

    loaded: list[type[AbstractCapability[Any]]] = []
    for name in sorted(required_names):
        providers = [entry_point for entry_point in entry_points if entry_point.name == name]
        if len(providers) > 1:
            raise ValueError(f"Capability {name!r} has multiple installed providers")
        if len(providers) == 1:
            capability_type = providers[0].load()
            if not isinstance(capability_type, type) or not issubclass(
                capability_type, AbstractCapability
            ):
                raise TypeError(
                    f"Capability provider {name!r} did not load an AbstractCapability class"
                )
            if capability_type.get_serialization_name() != name:
                raise ValueError(
                    f"Capability provider {name!r} loaded a class with a different name"
                )
            loaded.append(capability_type)

    return tuple(loaded)


def _find_names(value: Any, available_names: set[str]) -> set[str]:
    """Find installed capability names nested in serialized capability arguments."""
    found: set[str] = set()
    if isinstance(value, str):
        if value in available_names:
            found.add(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in available_names:
                found.add(key)
            found.update(_find_names(item, available_names))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_find_names(item, available_names))
    return found
