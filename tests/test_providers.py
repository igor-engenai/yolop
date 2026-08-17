from collections.abc import AsyncIterator
from importlib import metadata
from types import SimpleNamespace

from pydantic_ai import AgentRunResultEvent, AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import MonkeyPatch, raises

from yolop import Yolop


async def test_agent_spec_resolves_an_installed_model_provider(monkeypatch: MonkeyPatch) -> None:
    resolved: list[str] = []

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "provider model works"

    def resolve(model_name: str) -> FunctionModel:
        resolved.append(model_name)
        return FunctionModel(stream_function=respond)

    entry_point = SimpleNamespace(
        name="example",
        value="tests.test_providers:resolve",
        load=lambda: resolve,
    )
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: (entry_point,))

    async with Yolop().run(
        AgentSpec(model="example:model-a"),
        "Use the selected provider",
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "provider model works"
    assert resolved == ["model-a"]


async def test_explicit_model_reference_resolves_an_installed_provider(
    monkeypatch: MonkeyPatch,
) -> None:
    resolved: list[str] = []

    async def respond(
        _messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "explicit provider works"

    def resolve(model_name: str) -> FunctionModel:
        resolved.append(model_name)
        return FunctionModel(stream_function=respond)

    entry_point = SimpleNamespace(name="example", load=lambda: resolve)
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: (entry_point,))

    async with Yolop().run(
        AgentSpec(model="native:unused"),
        "Use the override",
        model="example:override",
        deps=None,
        deps_type=type(None),
    ) as run:
        async for _event in run:
            pass

    assert run.result is not None
    assert run.result.output == "explicit provider works"
    assert resolved == ["override"]


async def test_native_pydantic_model_reference_remains_native() -> None:
    async with Yolop().run(
        AgentSpec(model="test"),
        "Use Pydantic AI directly",
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)


def test_model_provider_requires_a_model_name(monkeypatch: MonkeyPatch) -> None:
    entry_point = SimpleNamespace(name="example", load=lambda: lambda _name: None)
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: (entry_point,))

    with raises(ValueError, match="Model provider 'example' requires a model name"):
        Yolop().run(
            AgentSpec(model="example:"),
            "Invalid model",
            deps=None,
            deps_type=type(None),
        )


def test_duplicate_model_provider_resolvers_are_rejected(monkeypatch: MonkeyPatch) -> None:
    entry_points = (
        SimpleNamespace(name="example", load=lambda: lambda _name: None),
        SimpleNamespace(name="example", load=lambda: lambda _name: None),
    )
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: entry_points)

    with raises(ValueError, match="Model provider 'example' has multiple installed resolvers"):
        Yolop().run(
            AgentSpec(model="example:model"),
            "Ambiguous model",
            deps=None,
            deps_type=type(None),
        )


def test_model_provider_must_load_a_callable(monkeypatch: MonkeyPatch) -> None:
    entry_point = SimpleNamespace(name="example", load=lambda: object())
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: (entry_point,))

    with raises(TypeError, match="Model provider 'example' did not load a callable resolver"):
        Yolop().run(
            AgentSpec(model="example:model"),
            "Invalid provider",
            deps=None,
            deps_type=type(None),
        )


def test_model_provider_must_resolve_a_native_model(monkeypatch: MonkeyPatch) -> None:
    entry_point = SimpleNamespace(name="example", load=lambda: lambda _name: object())
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: (entry_point,))

    with raises(TypeError, match="Model provider 'example' did not resolve a Pydantic AI Model"):
        Yolop().run(
            AgentSpec(model="example:model"),
            "Invalid model",
            deps=None,
            deps_type=type(None),
        )
