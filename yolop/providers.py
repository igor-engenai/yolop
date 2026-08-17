from pydantic_ai.models import Model

from .catalog import ProviderCatalog


def resolve_model_reference(reference: str, *, catalog: ProviderCatalog) -> Model | str:
    """Resolve a model reference through the runtime's immutable provider catalog."""
    return catalog.resolve_model_reference(reference)
