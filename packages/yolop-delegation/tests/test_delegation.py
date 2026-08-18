from __future__ import annotations

from pydantic_ai import AgentSpec
from pytest import raises
from yolop_delegation import (
    DelegateCatalog,
    DelegateConfigurationError,
    DelegateDefinition,
    DelegatePinMismatchError,
    DelegatePolicyError,
    DelegateUnknownError,
    bounded_idempotency_key,
    selections_from_spec,
)


def definition(
    alias: str = "research",
    *,
    version: str = "2026-08-18",
    model_id: str = "openai:gpt-5",
    max_depth: int = 2,
    max_children: int = 3,
) -> DelegateDefinition:
    return DelegateDefinition.from_spec(
        alias=alias,
        version=version,
        spec=AgentSpec(
            name=f"{alias}-agent",
            instructions="Return concise research notes.",
        ),
        model_id=model_id,
        max_depth=max_depth,
        max_children=max_children,
    )


def test_allowed_delegate_resolves_to_an_immutable_pin_and_manifest() -> None:
    catalog = DelegateCatalog({"tenant/acme": [definition()]})
    spec = AgentSpec(metadata={"delegation": {"delegates": [{"alias": "research"}]}})

    resolution = catalog.resolve_for_spec("tenant/acme", spec)

    assert resolution.aliases == ("research",)
    selected = resolution.selected[0]
    assert selected.alias == "research"
    assert selected.version == "2026-08-18"
    assert selected.model_id == "openai:gpt-5"
    assert selected.pin.digest == selected.digest
    assert selected.manifest == {
        "alias": "research",
        "version": "2026-08-18",
        "digest": selected.digest,
        "model_id": "openai:gpt-5",
        "max_depth": 2,
        "max_children": 3,
    }


def test_resolved_spec_is_a_fresh_snapshot() -> None:
    catalog = DelegateCatalog({"tenant/acme": [definition()]})
    selected = catalog.resolve("tenant/acme", "research")

    first = selected.spec
    first.name = "mutated-local-copy"

    assert selected.spec.name == "research-agent"


def test_unknown_and_forbidden_delegates_fail_without_cross_namespace_lookup() -> None:
    catalog = DelegateCatalog(
        {
            "tenant/acme": [definition(), definition(alias="allowed")],
            "tenant/beta": [definition(alias="private")],
        }
    )
    unknown = AgentSpec(metadata={"delegation": {"delegates": [{"alias": "private"}]}})
    parent_without_allowed = AgentSpec(
        metadata={"delegation": {"delegates": [{"alias": "research"}]}}
    )

    with raises(DelegateUnknownError):
        catalog.resolve_for_spec("tenant/acme", unknown)
    with raises(DelegatePolicyError, match="selected"):
        catalog.resolve_for_invocation("tenant/acme", parent_without_allowed, "allowed")
    with raises(DelegateUnknownError):
        catalog.resolve("tenant/missing", "research")


def test_parent_selection_cannot_expand_host_limits() -> None:
    catalog = DelegateCatalog({"tenant/acme": [definition(max_depth=2, max_children=3)]})

    with raises(DelegatePolicyError, match="max_depth"):
        catalog.resolve_for_spec(
            "tenant/acme",
            AgentSpec(
                metadata={"delegation": {"delegates": [{"alias": "research", "max_depth": 3}]}}
            ),
        )
    with raises(DelegatePolicyError, match="max_children"):
        catalog.resolve_for_spec(
            "tenant/acme",
            AgentSpec(
                metadata={"delegation": {"delegates": [{"alias": "research", "max_children": 4}]}}
            ),
        )


def test_depth_and_child_count_are_bounded_before_child_creation() -> None:
    selected = DelegateCatalog({"tenant/acme": [definition(max_depth=2, max_children=3)]}).resolve(
        "tenant/acme", "research"
    )

    selected.validate_invocation(depth=0, child_count=0)
    selected.validate_invocation(depth=1, child_count=2)

    with raises(DelegatePolicyError, match="depth"):
        selected.validate_invocation(depth=2, child_count=0)
    with raises(DelegatePolicyError, match="children"):
        selected.validate_invocation(depth=0, child_count=3)


def test_changed_digest_under_the_same_identity_is_rejected() -> None:
    original = definition()
    catalog = DelegateCatalog({"tenant/acme": [original]})
    changed = DelegateDefinition.from_spec(
        alias=original.alias,
        version=original.version,
        spec=AgentSpec(name="changed", instructions="Changed instructions."),
        model_id=original.model_id,
        max_depth=original.max_depth,
        max_children=original.max_children,
    )

    with raises(DelegatePinMismatchError):
        DelegateCatalog({"tenant/acme": [changed]}).resolve_pin("tenant/acme", original.pin)
    assert catalog.resolve_pin("tenant/acme", original.pin).digest == original.digest


def test_duplicate_alias_and_invalid_metadata_are_rejected() -> None:
    with raises(DelegateConfigurationError, match="unique"):
        DelegateCatalog({"tenant/acme": [definition(), definition(version="2026-08-19")]})

    with raises(DelegateConfigurationError):
        selections_from_spec(
            AgentSpec(metadata={"delegation": {"delegates": [{"alias": "bad name"}]}})
        )


def test_generated_idempotency_keys_are_stable_and_runtime_safe() -> None:
    first = bounded_idempotency_key("delegate", "x" * 10_000)

    assert first == bounded_idempotency_key("delegate", "x" * 10_000)
    assert len(first) <= 255


def test_parent_selection_limits_can_be_tighter_than_host_limits() -> None:
    catalog = DelegateCatalog({"tenant/acme": [definition(max_depth=3, max_children=4)]})
    spec = AgentSpec(
        metadata={
            "delegation": {"delegates": [{"alias": "research", "max_depth": 1, "max_children": 2}]}
        }
    )

    selected = catalog.resolve_for_spec("tenant/acme", spec).selected[0]

    assert selected.max_depth == 1
    assert selected.max_children == 2
