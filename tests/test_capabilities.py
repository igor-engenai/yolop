from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from pydantic_ai import AgentRunResultEvent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import raises

from yolop import ProviderCatalog, Yolop


@dataclass
class Greeting(AbstractCapability[None]):
    text: str = ""

    def get_instructions(self) -> str:
        return self.text


def test_unallowlisted_capability_is_rejected_without_loading_it() -> None:
    loaded: list[str] = []

    def load_forbidden() -> type[Greeting]:
        loaded.append("forbidden")
        return Greeting

    entry_point = SimpleNamespace(
        name="Forbidden",
        value="tests:Forbidden",
        load=load_forbidden,
    )
    catalog = ProviderCatalog.from_entry_points(
        capability_entry_points=(entry_point,),
        allowed_capabilities=(),
    )

    with raises(ValueError, match="Capability 'Forbidden' is not in provider catalog"):
        Yolop(provider_catalog=catalog).run(
            {"capabilities": ["Forbidden"]},
            "Must fail before loading",
            model=FunctionModel(stream_function=_unused_stream),
            deps=None,
            deps_type=type(None),
        )

    assert loaded == []


async def test_agent_spec_loads_its_custom_capability_plugin() -> None:
    loaded: list[str] = []

    def load() -> type[Greeting]:
        loaded.append("Greeting")
        return Greeting

    def load_unused() -> type[Greeting]:
        loaded.append("Unused")
        return Greeting

    entry_points = (
        SimpleNamespace(name="Greeting", value="tests:Greeting", load=load),
        SimpleNamespace(name="Unused", value="tests:Unused", load=load_unused),
    )
    catalog = ProviderCatalog.from_entry_points(
        capability_entry_points=entry_points,
        allowed_capabilities={"Greeting"},
    )

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        assert info.instructions == "Loaded from the capability plugin"
        yield "done"

    runtime = Yolop(provider_catalog=catalog)
    spec: dict[str, Any] = {
        "capabilities": [
            {"Greeting": {"text": "Loaded from the capability plugin"}},
        ]
    }

    async with runtime.run(
        spec,
        "Use the plugin",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    assert loaded == ["Greeting"]


async def test_runtime_reuses_one_catalog_without_per_run_discovery() -> None:
    loaded: list[str] = []

    def load() -> type[Greeting]:
        loaded.append("Greeting")
        return Greeting

    catalog = ProviderCatalog.from_entry_points(
        capability_entry_points=(
            SimpleNamespace(name="Greeting", value="tests:Greeting", load=load),
        ),
        allowed_capabilities={"Greeting"},
    )
    runtime = Yolop(provider_catalog=catalog)

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        assert info.instructions == "Reusable catalog"
        yield "done"

    spec = {"capabilities": [{"Greeting": {"text": "Reusable catalog"}}]}

    for _ in range(2):
        async with runtime.run(
            spec,
            "Use the catalog",
            model=FunctionModel(stream_function=respond),
            deps=None,
            deps_type=type(None),
        ) as run:
            events = [event async for event in run]
        assert isinstance(events[-1], AgentRunResultEvent)

    assert loaded == ["Greeting"]


async def test_nested_custom_capability_is_loaded() -> None:
    entry_point = SimpleNamespace(name="Greeting", value="tests:Greeting", load=lambda: Greeting)
    catalog = ProviderCatalog.from_entry_points(
        capability_entry_points=(entry_point,),
        allowed_capabilities={"Greeting"},
    )

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        assert info.instructions == "Nested capability loaded"
        yield "done"

    spec: dict[str, Any] = {
        "capabilities": [
            {
                "PrefixTools": {
                    "prefix": "nested",
                    "capability": {"Greeting": {"text": "Nested capability loaded"}},
                }
            }
        ]
    }

    async with Yolop(provider_catalog=catalog).run(
        spec,
        "Use the nested plugin",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)


def test_unknown_custom_capability_fails_before_execution() -> None:
    with raises(ValueError, match="Capability 'Missing' is not in provider catalog"):
        Yolop(provider_catalog=ProviderCatalog()).run(
            {"capabilities": ["Missing"]},
            "Use the plugin",
            model=FunctionModel(stream_function=_unused_stream),
            deps=None,
            deps_type=type(None),
        )


def test_selected_capability_rejects_duplicate_providers() -> None:
    entry_points = (
        SimpleNamespace(name="Greeting", value="first:Greeting", load=lambda: Greeting),
        SimpleNamespace(name="Greeting", value="second:Greeting", load=lambda: Greeting),
    )
    with raises(ValueError, match="Capability provider 'Greeting' has multiple owners"):
        Yolop(
            provider_catalog=ProviderCatalog.from_entry_points(
                capability_entry_points=entry_points,
            )
        ).run(
            {"capabilities": ["Greeting"]},
            "Use the plugin",
            model=FunctionModel(stream_function=_unused_stream),
            deps=None,
            deps_type=type(None),
        )


async def _unused_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    yield "unused"
