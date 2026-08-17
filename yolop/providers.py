from collections.abc import Callable
from importlib import metadata

from pydantic_ai.models import Model

_ENTRY_POINT_GROUP = "yolop.model_providers"

ModelResolver = Callable[[str], Model]


def resolve_model_reference(reference: str) -> Model | str:
    """Resolve an installed YoloP model provider, or preserve a native reference."""
    provider_name, separator, model_name = reference.partition(":")
    if not separator:
        return reference

    entry_points = metadata.entry_points(group=_ENTRY_POINT_GROUP)
    providers = [entry_point for entry_point in entry_points if entry_point.name == provider_name]
    if not providers:
        return reference
    if len(providers) > 1:
        raise ValueError(f"Model provider {provider_name!r} has multiple installed resolvers")
    if not model_name:
        raise ValueError(f"Model provider {provider_name!r} requires a model name")

    resolver = providers[0].load()
    if not callable(resolver):
        raise TypeError(f"Model provider {provider_name!r} did not load a callable resolver")
    model = resolver(model_name)
    if not isinstance(model, Model):
        raise TypeError(f"Model provider {provider_name!r} did not resolve a Pydantic AI Model")
    return model
