from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from pydantic_ai import AgentSpec
from pydantic_ai.capabilities import CAPABILITY_TYPES, AbstractCapability
from pydantic_ai.models import Model

_CAPABILITY_ENTRY_POINT_GROUP = "yolop.capabilities"
_MODEL_ENTRY_POINT_GROUP = "yolop.model_providers"
_NESTED_CAPABILITY_KEYS = frozenset({"capability", "capabilities", "wrapped"})


@dataclass(frozen=True)
class ProviderManifest:
    """Stable identity information for one loaded provider entry point."""

    group: str
    name: str
    distribution: str | None
    version: str | None
    identity: str


@dataclass(frozen=True)
class CapabilityProvider:
    """One loaded custom capability provider."""

    name: str
    capability_type: type[AbstractCapability[Any]]
    manifest: ProviderManifest


@dataclass(frozen=True)
class ModelProvider:
    """One loaded model resolver provider."""

    name: str
    resolver: Any
    manifest: ProviderManifest


@dataclass(frozen=True)
class ProviderCatalog:
    """Immutable capability and model-provider definitions for a runtime."""

    capabilities: tuple[CapabilityProvider, ...] = ()
    model_providers: tuple[ModelProvider, ...] = ()
    blocked_model_providers: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        capabilities = tuple(sorted(self.capabilities, key=lambda provider: provider.name))
        model_providers = tuple(sorted(self.model_providers, key=lambda provider: provider.name))
        _ensure_unique("capability", capabilities)
        _ensure_unique("model provider", model_providers)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "model_providers", model_providers)
        object.__setattr__(self, "blocked_model_providers", frozenset(self.blocked_model_providers))

    @classmethod
    def from_installed(
        cls,
        *,
        allowed_capabilities: Collection[str] | None = None,
        allowed_model_providers: Collection[str] | None = None,
    ) -> ProviderCatalog:
        """Build a catalog from installed entry points and optional allowlists."""
        return cls.from_entry_points(
            capability_entry_points=metadata.entry_points(group=_CAPABILITY_ENTRY_POINT_GROUP),
            model_provider_entry_points=metadata.entry_points(group=_MODEL_ENTRY_POINT_GROUP),
            allowed_capabilities=allowed_capabilities,
            allowed_model_providers=allowed_model_providers,
        )

    @classmethod
    def from_entry_points(
        cls,
        *,
        capability_entry_points: Iterable[Any] = (),
        model_provider_entry_points: Iterable[Any] = (),
        allowed_capabilities: Collection[str] | None = None,
        allowed_model_providers: Collection[str] | None = None,
    ) -> ProviderCatalog:
        """Build a catalog without consulting metadata.

        Entry points outside an allowlist are discarded before their ``load`` method is called.
        """
        all_capability_points = tuple(capability_entry_points)
        all_model_points = tuple(model_provider_entry_points)
        capability_points = _allowed_entry_points(all_capability_points, allowed_capabilities)
        model_points = _allowed_entry_points(all_model_points, allowed_model_providers)
        _ensure_unique_entry_points("capability", capability_points)
        _ensure_unique_entry_points("model provider", model_points)

        capabilities = tuple(_load_capability(entry_point) for entry_point in capability_points)
        model_providers = tuple(_load_model_provider(entry_point) for entry_point in model_points)
        blocked_model_providers = (
            frozenset(entry_point.name for entry_point in all_model_points)
            - frozenset(entry_point.name for entry_point in model_points)
            if allowed_model_providers is not None
            else frozenset()
        )
        return cls(
            capabilities=capabilities,
            model_providers=model_providers,
            blocked_model_providers=blocked_model_providers,
        )

    @property
    def manifest(self) -> tuple[ProviderManifest, ...]:
        """Return the sorted, inspectable provider manifest."""
        return tuple(provider.manifest for provider in (*self.capabilities, *self.model_providers))

    def has_capability(self, name: str) -> bool:
        return any(provider.name == name for provider in self.capabilities)

    def capability_type(self, name: str) -> type[AbstractCapability[Any]]:
        for provider in self.capabilities:
            if provider.name == name:
                return provider.capability_type
        raise ValueError(f"Capability {name!r} is not in provider catalog")

    def validate_spec(self, spec: AgentSpec | Mapping[str, Any]) -> None:
        """Reject custom capabilities unavailable in this catalog before AgentSpec loading."""
        for name in _selected_capability_names(spec):
            if name not in CAPABILITY_TYPES and not self.has_capability(name):
                raise ValueError(f"Capability {name!r} is not in provider catalog")

    def capability_types_for(
        self, spec: AgentSpec | Mapping[str, Any]
    ) -> tuple[type[AbstractCapability[Any]], ...]:
        """Return only the custom capability classes selected by ``spec``."""
        self.validate_spec(spec)
        selected_names = {
            name for name in _selected_capability_names(spec) if self.has_capability(name)
        }
        return tuple(self.capability_type(name) for name in sorted(selected_names))

    def resolve_model_reference(self, reference: str) -> Model | str:
        """Resolve a catalog model provider, or preserve a native Pydantic AI reference."""
        provider_name, separator, model_name = reference.partition(":")
        if separator and provider_name in self.blocked_model_providers:
            raise ValueError(f"Model provider {provider_name!r} is not in provider catalog")
        if not separator or not any(
            provider.name == provider_name for provider in self.model_providers
        ):
            return reference
        if not model_name:
            raise ValueError(f"Model provider {provider_name!r} requires a model name")

        resolver = next(
            provider.resolver for provider in self.model_providers if provider.name == provider_name
        )
        model = resolver(model_name)
        if not isinstance(model, Model):
            raise TypeError(f"Model provider {provider_name!r} did not resolve a Pydantic AI Model")
        return model


def _allowed_entry_points(
    entry_points: Iterable[Any], allowed_names: Collection[str] | None
) -> tuple[Any, ...]:
    allowed = None if allowed_names is None else frozenset(allowed_names)
    return tuple(
        entry_point
        for entry_point in entry_points
        if allowed is None or entry_point.name in allowed
    )


def _ensure_unique_entry_points(kind: str, entry_points: Iterable[Any]) -> None:
    grouped: dict[str, list[Any]] = {}
    for entry_point in entry_points:
        grouped.setdefault(entry_point.name, []).append(entry_point)
    for name, providers in grouped.items():
        if len(providers) > 1:
            owners = ", ".join(_entry_point_owner(provider) for provider in providers)
            raise ValueError(f"{_provider_label(kind)} {name!r} has multiple owners: {owners}")


def _ensure_unique(kind: str, providers: Iterable[Any]) -> None:
    names: set[str] = set()
    for provider in providers:
        if provider.name in names:
            raise ValueError(f"{_provider_label(kind)} {provider.name!r} has multiple owners")
        names.add(provider.name)


def _load_capability(entry_point: Any) -> CapabilityProvider:
    capability_type = entry_point.load()
    if not isinstance(capability_type, type) or not issubclass(capability_type, AbstractCapability):
        raise TypeError(
            f"Capability provider {entry_point.name!r} did not load an AbstractCapability class"
        )
    if capability_type.get_serialization_name() != entry_point.name:
        raise ValueError(
            f"Capability provider {entry_point.name!r} loaded a class with a different name"
        )
    return CapabilityProvider(
        name=entry_point.name,
        capability_type=capability_type,
        manifest=_manifest(_CAPABILITY_ENTRY_POINT_GROUP, entry_point, capability_type),
    )


def _load_model_provider(entry_point: Any) -> ModelProvider:
    resolver = entry_point.load()
    if not callable(resolver):
        raise TypeError(f"Model provider {entry_point.name!r} did not load a callable resolver")
    return ModelProvider(
        name=entry_point.name,
        resolver=resolver,
        manifest=_manifest(_MODEL_ENTRY_POINT_GROUP, entry_point, resolver),
    )


def _manifest(group: str, entry_point: Any, loaded: Any) -> ProviderManifest:
    distribution = getattr(getattr(entry_point, "dist", None), "name", None)
    version = getattr(getattr(entry_point, "dist", None), "version", None)
    return ProviderManifest(
        group=group,
        name=entry_point.name,
        distribution=distribution,
        version=version,
        identity=_identity(loaded),
    )


def _entry_point_owner(entry_point: Any) -> str:
    distribution = getattr(getattr(entry_point, "dist", None), "name", None)
    version = getattr(getattr(entry_point, "dist", None), "version", None)
    owner = distribution or "unknown distribution"
    if version:
        owner = f"{owner} {version}"
    target = getattr(entry_point, "value", "unknown target")
    return f"{owner} ({target})"


def _provider_label(kind: str) -> str:
    return "Capability provider" if kind == "capability" else "Model provider"


def _identity(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}:{qualname}"


def _selected_capability_names(spec: AgentSpec | Mapping[str, Any]) -> tuple[str, ...]:
    if isinstance(spec, AgentSpec):
        capabilities = spec.capabilities
        names: list[str] = []
        for capability in capabilities:
            names.append(capability.name)
            names.extend(_nested_capability_names(capability.arguments))
        return tuple(names)

    capabilities = spec.get("capabilities", ())
    names = []
    for capability in capabilities or ():
        if isinstance(capability, str):
            names.append(capability)
        elif isinstance(capability, Mapping):
            for name, arguments in capability.items():
                names.append(name)
                names.extend(_nested_capability_names(arguments))
    return tuple(names)


def _nested_capability_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _NESTED_CAPABILITY_KEYS and isinstance(item, Mapping):
                for name, arguments in item.items():
                    names.append(name)
                    names.extend(_nested_capability_names(arguments))
            elif key in _NESTED_CAPABILITY_KEYS and isinstance(item, (list, tuple)):
                for nested in item:
                    if isinstance(nested, str):
                        names.append(nested)
                    elif isinstance(nested, Mapping):
                        for name, arguments in nested.items():
                            names.append(name)
                            names.extend(_nested_capability_names(arguments))
    return names


__all__ = [
    "CapabilityProvider",
    "ModelProvider",
    "ProviderCatalog",
    "ProviderManifest",
]
