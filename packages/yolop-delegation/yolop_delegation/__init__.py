"""Host-authorized immutable delegate AgentSpecs for YoloP."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import AgentSpec
from yolop_runtime import agent_spec_digest, validate_namespace

from yolop import ProviderCatalog

_ALIAS_PATTERN = r"^[A-Za-z][A-Za-z0-9-]{0,63}$"
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class DelegateConfigurationError(ValueError):
    """Host delegate configuration is invalid."""

    code = "delegate_configuration_error"


class DelegateUnknownError(ValueError):
    """A delegate is not available in the requested namespace."""

    code = "delegate_unknown"


class DelegatePolicyError(ValueError):
    """A parent selection or invocation exceeds host policy."""

    code = "delegate_policy_error"


class DelegatePinMismatchError(ValueError):
    """A persisted delegate pin no longer matches the host catalog."""

    code = "delegate_pin_mismatch"


class DelegateSelection(BaseModel):
    """Safe parent-AgentSpec selection for one host delegate alias."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(pattern=_ALIAS_PATTERN)
    max_depth: int | None = Field(default=None, ge=1, le=64)
    max_children: int | None = Field(default=None, ge=1, le=1024)


class DelegateAgentSelection(BaseModel):
    """The ``metadata.delegation`` section of an AgentSpec."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    delegates: tuple[DelegateSelection, ...] = ()

    @model_validator(mode="after")
    def validate_aliases(self) -> DelegateAgentSelection:
        aliases = [selection.alias for selection in self.delegates]
        if len(aliases) != len(set(aliases)):
            raise DelegateConfigurationError(
                "Delegate aliases must be unique in AgentSpec metadata"
            )
        return self


@dataclass(frozen=True)
class DelegatePin:
    """Immutable identity used to verify a delegate across execution boundaries."""

    alias: str
    version: str
    digest: str
    model_id: str


@dataclass(frozen=True)
class DelegateDefinition:
    """Host-owned immutable AgentSpec and model policy for one delegate alias."""

    alias: str
    version: str
    model_id: str
    max_depth: int
    max_children: int
    digest: str
    _spec_json: str = field(repr=False)

    @classmethod
    def from_spec(
        cls,
        *,
        alias: str,
        version: str,
        spec: AgentSpec | Mapping[str, Any],
        model_id: str,
        max_depth: int = 1,
        max_children: int = 1,
    ) -> DelegateDefinition:
        """Create a definition from a host-owned, canonical AgentSpec snapshot."""
        try:
            validated = spec if isinstance(spec, AgentSpec) else AgentSpec.model_validate(spec)
            payload = validated.model_dump(mode="json")
            spec_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise DelegateConfigurationError(
                "Delegate AgentSpec is not JSON-serializable"
            ) from error
        return cls(
            alias=alias,
            version=version,
            model_id=model_id,
            max_depth=max_depth,
            max_children=max_children,
            digest=agent_spec_digest(validated),
            _spec_json=spec_json,
        )

    def __post_init__(self) -> None:
        if re.fullmatch(_ALIAS_PATTERN, self.alias) is None:
            raise DelegateConfigurationError("Delegate alias has an invalid format")
        if _VERSION_PATTERN.fullmatch(self.version) is None:
            raise DelegateConfigurationError("Delegate version has an invalid format")
        if not self.model_id.strip():
            raise DelegateConfigurationError("Delegate model ID must not be empty")
        if isinstance(self.max_depth, bool) or not 1 <= self.max_depth <= 64:
            raise DelegateConfigurationError("Delegate max_depth must be between 1 and 64")
        if isinstance(self.max_children, bool) or not 1 <= self.max_children <= 1024:
            raise DelegateConfigurationError("Delegate max_children must be between 1 and 1024")
        try:
            spec = AgentSpec.model_validate(json.loads(self._spec_json))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise DelegateConfigurationError("Delegate AgentSpec snapshot is invalid") from error
        if agent_spec_digest(spec) != self.digest:
            raise DelegateConfigurationError(
                "Delegate AgentSpec digest does not match its snapshot"
            )

    @property
    def spec(self) -> AgentSpec:
        """Return a fresh AgentSpec copy so callers cannot mutate the catalog."""
        return AgentSpec.model_validate(json.loads(self._spec_json))

    @property
    def pin(self) -> DelegatePin:
        """Return the durable identity for this definition."""
        return DelegatePin(
            alias=self.alias,
            version=self.version,
            digest=self.digest,
            model_id=self.model_id,
        )


@dataclass(frozen=True)
class ResolvedDelegate:
    """One parent-authorized delegate with effective bounded limits."""

    definition: DelegateDefinition
    max_depth: int
    max_children: int

    @property
    def alias(self) -> str:
        return self.definition.alias

    @property
    def version(self) -> str:
        return self.definition.version

    @property
    def model_id(self) -> str:
        return self.definition.model_id

    @property
    def digest(self) -> str:
        return self.definition.digest

    @property
    def pin(self) -> DelegatePin:
        return self.definition.pin

    @property
    def spec(self) -> AgentSpec:
        return self.definition.spec

    @property
    def manifest(self) -> dict[str, Any]:
        """Return safe resolution metadata without AgentSpec contents."""
        return {
            "alias": self.alias,
            "version": self.version,
            "digest": self.digest,
            "model_id": self.model_id,
            "max_depth": self.max_depth,
            "max_children": self.max_children,
        }

    def validate_invocation(self, *, depth: int, child_count: int) -> None:
        """Validate ancestry limits before creating a child Session or Run."""
        if isinstance(depth, bool) or depth < 0:
            raise DelegatePolicyError("Delegate depth must be non-negative")
        if isinstance(child_count, bool) or child_count < 0:
            raise DelegatePolicyError("Delegate child count must be non-negative")
        if depth >= self.max_depth:
            raise DelegatePolicyError("Delegate maximum depth has been reached")
        if child_count >= self.max_children:
            raise DelegatePolicyError("Delegate maximum children has been reached")


@dataclass(frozen=True)
class DelegateResolution:
    """Resolved parent selections for one namespace."""

    namespace: str
    selected: tuple[ResolvedDelegate, ...]

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(delegate.alias for delegate in self.selected)

    @property
    def manifest(self) -> tuple[dict[str, Any], ...]:
        return tuple(delegate.manifest for delegate in self.selected)

    def for_alias(self, alias: str) -> ResolvedDelegate:
        for delegate in self.selected:
            if delegate.alias == alias:
                return delegate
        raise DelegatePolicyError(
            f"Delegate alias {alias!r} is not selected by the parent AgentSpec"
        )


class DelegateCatalog:
    """Immutable host catalog partitioned by namespace."""

    def __init__(
        self,
        definitions_by_namespace: Mapping[str, Iterable[DelegateDefinition]],
        *,
        provider_catalog: ProviderCatalog | None = None,
    ) -> None:
        if not isinstance(definitions_by_namespace, Mapping):
            raise DelegateConfigurationError("Delegate catalog must be a namespace mapping")
        namespaces: dict[str, Mapping[str, DelegateDefinition]] = {}
        for namespace, definitions in definitions_by_namespace.items():
            try:
                validated_namespace = validate_namespace(namespace)
            except ValueError as error:
                raise DelegateConfigurationError("Delegate namespace is invalid") from error
            by_alias: dict[str, DelegateDefinition] = {}
            try:
                definitions_tuple = tuple(definitions)
            except TypeError as error:
                raise DelegateConfigurationError("Delegate definitions must be iterable") from error
            for definition in definitions_tuple:
                if not isinstance(definition, DelegateDefinition):
                    raise DelegateConfigurationError(
                        "Delegate catalog contains an invalid definition"
                    )
                if provider_catalog is not None:
                    provider_catalog.validate_spec(definition.spec)
                if definition.alias in by_alias:
                    raise DelegateConfigurationError(
                        f"Delegate aliases must be unique in namespace {validated_namespace!r}"
                    )
                by_alias[definition.alias] = definition
            namespaces[validated_namespace] = MappingProxyType(by_alias)
        self._namespaces = MappingProxyType(namespaces)

    def aliases(self, namespace: str) -> tuple[str, ...]:
        """Return aliases visible in one namespace."""
        return tuple(sorted(self._namespace(namespace)))

    def resolve(self, namespace: str, alias: str) -> ResolvedDelegate:
        """Resolve one host-authorized alias without crossing namespace boundaries."""
        definition = self._namespace(namespace).get(alias)
        if definition is None:
            raise DelegateUnknownError("Delegate alias is not available in this namespace")
        return ResolvedDelegate(
            definition=definition,
            max_depth=definition.max_depth,
            max_children=definition.max_children,
        )

    def resolve_for_spec(
        self,
        namespace: str,
        spec: AgentSpec | Mapping[str, Any],
    ) -> DelegateResolution:
        """Resolve every alias selected by an AgentSpec before child execution."""
        selections = selections_from_spec(spec)
        selected: list[ResolvedDelegate] = []
        for selection in selections:
            delegate = self.resolve(namespace, selection.alias)
            selected.append(
                ResolvedDelegate(
                    definition=delegate.definition,
                    max_depth=_effective_limit(
                        selection.max_depth,
                        delegate.max_depth,
                        field_name="max_depth",
                    ),
                    max_children=_effective_limit(
                        selection.max_children,
                        delegate.max_children,
                        field_name="max_children",
                    ),
                )
            )
        return DelegateResolution(namespace=namespace, selected=tuple(selected))

    def resolve_for_invocation(
        self,
        namespace: str,
        parent_spec: AgentSpec | Mapping[str, Any],
        alias: str,
    ) -> ResolvedDelegate:
        """Resolve an alias only when the parent AgentSpec explicitly selected it."""
        return self.resolve_for_spec(namespace, parent_spec).for_alias(alias)

    def resolve_pin(self, namespace: str, pin: DelegatePin) -> ResolvedDelegate:
        """Resolve a persisted pin and reject replacement under the same identity."""
        current = self.resolve(namespace, pin.alias)
        if current.pin != pin:
            raise DelegatePinMismatchError("Delegate pin no longer matches the host catalog")
        return current

    def _namespace(self, namespace: str) -> Mapping[str, DelegateDefinition]:
        try:
            validated_namespace = validate_namespace(namespace)
        except ValueError as error:
            raise DelegateUnknownError("Delegate namespace is not available") from error
        definitions = self._namespaces.get(validated_namespace)
        if definitions is None:
            raise DelegateUnknownError("Delegate namespace is not available")
        return definitions


def selections_from_spec(spec: object) -> tuple[DelegateSelection, ...]:
    """Read safe delegate alias selections from AgentSpec metadata."""
    if isinstance(spec, Mapping):
        raw_metadata = spec.get("metadata")
    else:
        raw_metadata = getattr(spec, "metadata", None)
    if raw_metadata is None:
        return ()
    if not isinstance(raw_metadata, Mapping):
        raise DelegateConfigurationError("AgentSpec metadata must be an object")
    raw_delegation = raw_metadata.get("delegation")
    if raw_delegation is None:
        return ()
    try:
        selection = DelegateAgentSelection.model_validate(raw_delegation)
    except DelegateConfigurationError:
        raise
    except Exception as error:
        raise DelegateConfigurationError("AgentSpec delegation metadata is invalid") from error
    return selection.delegates


def _effective_limit(requested: int | None, host_limit: int, *, field_name: str) -> int:
    if requested is None:
        return host_limit
    if requested > host_limit:
        raise DelegatePolicyError(f"Delegate parent selection {field_name} exceeds the host limit")
    return requested


__all__ = [
    "DelegateAgentSelection",
    "DelegateCatalog",
    "DelegateConfigurationError",
    "DelegateDefinition",
    "DelegatePin",
    "DelegatePinMismatchError",
    "DelegatePolicyError",
    "DelegateResolution",
    "DelegateSelection",
    "DelegateUnknownError",
    "ResolvedDelegate",
    "selections_from_spec",
]
