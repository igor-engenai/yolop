from collections.abc import AsyncIterator
from types import SimpleNamespace

from pydantic_ai import AgentRunResultEvent, AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import raises

from yolop import ProviderCatalog, Yolop


def test_unallowlisted_model_provider_is_rejected_without_loading_it() -> None:
    loaded: list[str] = []

    def load_forbidden():
        loaded.append("forbidden")
        return lambda _model_name: object()

    entry_point = SimpleNamespace(name="forbidden", load=load_forbidden)
    catalog = ProviderCatalog.from_entry_points(
        model_provider_entry_points=(entry_point,),
        allowed_model_providers=(),
    )

    with raises(ValueError, match="Model provider 'forbidden' is not in provider catalog"):
        Yolop(provider_catalog=catalog).run(
            AgentSpec(model="forbidden:model"),
            "Must fail before loading",
            deps=None,
            deps_type=type(None),
        )

    assert loaded == []


async def test_agent_spec_resolves_an_installed_model_provider() -> None:
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
    catalog = ProviderCatalog.from_entry_points(model_provider_entry_points=(entry_point,))

    async with Yolop(provider_catalog=catalog).run(
        AgentSpec(model="example:model-a"),
        "Use the selected provider",
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "provider model works"
    assert resolved == ["model-a"]


async def test_explicit_model_reference_resolves_an_installed_provider() -> None:
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
    catalog = ProviderCatalog.from_entry_points(model_provider_entry_points=(entry_point,))

    async with Yolop(provider_catalog=catalog).run(
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
    async with Yolop(provider_catalog=ProviderCatalog()).run(
        AgentSpec(model="test"),
        "Use Pydantic AI directly",
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)


def test_model_provider_requires_a_model_name() -> None:
    entry_point = SimpleNamespace(name="example", load=lambda: lambda _name: None)
    catalog = ProviderCatalog.from_entry_points(model_provider_entry_points=(entry_point,))

    with raises(ValueError, match="Model provider 'example' requires a model name"):
        Yolop(provider_catalog=catalog).run(
            AgentSpec(model="example:"),
            "Invalid model",
            deps=None,
            deps_type=type(None),
        )


def test_duplicate_model_provider_resolvers_are_rejected() -> None:
    entry_points = (
        SimpleNamespace(name="example", value="first:resolve", load=lambda: lambda _name: None),
        SimpleNamespace(name="example", value="second:resolve", load=lambda: lambda _name: None),
    )

    with raises(ValueError, match="Model provider 'example' has multiple owners"):
        ProviderCatalog.from_entry_points(model_provider_entry_points=entry_points)


def test_model_provider_must_load_a_callable() -> None:
    entry_point = SimpleNamespace(name="example", load=lambda: object())

    with raises(TypeError, match="Model provider 'example' did not load a callable resolver"):
        ProviderCatalog.from_entry_points(model_provider_entry_points=(entry_point,))


def test_model_provider_must_resolve_a_native_model() -> None:
    entry_point = SimpleNamespace(name="example", load=lambda: lambda _name: object())
    catalog = ProviderCatalog.from_entry_points(model_provider_entry_points=(entry_point,))

    with raises(TypeError, match="Model provider 'example' did not resolve a Pydantic AI Model"):
        Yolop(provider_catalog=catalog).run(
            AgentSpec(model="example:model"),
            "Invalid model",
            deps=None,
            deps_type=type(None),
        )
