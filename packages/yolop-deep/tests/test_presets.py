from __future__ import annotations

from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from yolop_deep import load_deep_coding_spec, load_deep_research_spec
from yolop_delegation import selections_from_spec

from yolop import ProviderCatalog, Yolop


def catalog() -> ProviderCatalog:
    return ProviderCatalog.from_installed()


def test_deep_presets_are_explicit_and_host_resource_selection_is_metadata() -> None:
    coding = load_deep_coding_spec(catalog=catalog())
    research = load_deep_research_spec(catalog=catalog())

    assert coding.name == "deep-coding"
    assert research.name == "deep-research"
    metadata = cast(dict[str, Any], research.metadata)
    assert tuple(selection.alias for selection in selections_from_spec(research)) == (
        "research",
        "review",
    )
    assert metadata["memory"]["scopes"] == ["session", "workspace"]
    assert all(capability.name != "Team" for capability in research.capabilities)


def test_deep_presets_construct_with_installed_capabilities() -> None:
    kernel = Yolop()

    for spec in (
        load_deep_coding_spec(catalog=catalog()),
        load_deep_research_spec(catalog=catalog()),
    ):
        capabilities = kernel.resolve_capabilities(spec)
        Agent.from_spec(
            spec,
            model=TestModel(),
            custom_capability_types=capabilities.selected_types,
            capabilities=capabilities.enforced_capabilities,
        )


def test_deep_coding_marks_forks_as_experimental() -> None:
    spec = load_deep_coding_spec(catalog=catalog())
    metadata = cast(dict[str, Any], spec.metadata)

    assert metadata["delegation"]["experimental_forks"] is True
    assert tuple(selection.alias for selection in selections_from_spec(spec)) == (
        "research",
        "review",
    )
