from collections.abc import AsyncIterator
from dataclasses import dataclass
from importlib import metadata
from types import SimpleNamespace
from typing import Any

from pydantic_ai import AgentRunResultEvent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import MonkeyPatch, raises

from yolop import Yolop


@dataclass
class Greeting(AbstractCapability[None]):
    text: str = ""

    def get_instructions(self) -> str:
        return self.text


async def test_agent_spec_loads_its_custom_capability_plugin(monkeypatch: MonkeyPatch) -> None:
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
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: entry_points)

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        assert info.instructions == "Loaded from the capability plugin"
        yield "done"

    runtime = Yolop()
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


async def test_nested_custom_capability_is_loaded(monkeypatch: MonkeyPatch) -> None:
    entry_point = SimpleNamespace(name="Greeting", value="tests:Greeting", load=lambda: Greeting)
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: (entry_point,))

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

    async with Yolop().run(
        spec,
        "Use the nested plugin",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)


def test_unknown_custom_capability_fails_before_execution(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: ())

    with raises(ValueError, match="Capability 'Missing'.*custom_capability_types"):
        Yolop().run(
            {"capabilities": ["Missing"]},
            "Use the plugin",
            model=FunctionModel(stream_function=_unused_stream),
            deps=None,
            deps_type=type(None),
        )


def test_selected_capability_rejects_duplicate_providers(monkeypatch: MonkeyPatch) -> None:
    entry_points = (
        SimpleNamespace(name="Greeting", value="first:Greeting", load=lambda: Greeting),
        SimpleNamespace(name="Greeting", value="second:Greeting", load=lambda: Greeting),
    )
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: entry_points)

    with raises(ValueError, match="Capability 'Greeting' has multiple installed providers"):
        Yolop().run(
            {"capabilities": ["Greeting"]},
            "Use the plugin",
            model=FunctionModel(stream_function=_unused_stream),
            deps=None,
            deps_type=type(None),
        )


async def _unused_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    yield "unused"
