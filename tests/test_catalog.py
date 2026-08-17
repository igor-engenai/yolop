from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace

from pydantic_ai.capabilities import AbstractCapability
from pytest import raises

from yolop import ProviderCatalog


@dataclass
class Greeting(AbstractCapability[None]):
    text: str = ""


def test_provider_catalog_has_a_stable_inspectable_manifest() -> None:
    capability_entry_point = SimpleNamespace(
        name="Greeting",
        value="tests.test_catalog:Greeting",
        dist=SimpleNamespace(name="yolop-example", version="1.2.3"),
        load=lambda: Greeting,
    )

    def resolver(_model_name: str) -> object:
        return object()

    model_entry_point = SimpleNamespace(
        name="example",
        value="tests.test_catalog:resolver",
        dist=SimpleNamespace(name="yolop-provider-example", version="4.5.6"),
        load=lambda: resolver,
    )

    catalog = ProviderCatalog.from_entry_points(
        capability_entry_points=(capability_entry_point,),
        model_provider_entry_points=(model_entry_point,),
    )

    assert catalog.manifest[0].group == "yolop.capabilities"
    assert catalog.manifest[0].name == "Greeting"
    assert catalog.manifest[0].distribution == "yolop-example"
    assert catalog.manifest[0].version == "1.2.3"
    assert catalog.manifest[0].identity == f"{Greeting.__module__}:{Greeting.__qualname__}"
    assert catalog.manifest[1].group == "yolop.model_providers"
    assert catalog.manifest[1].name == "example"
    assert catalog.manifest[1].distribution == "yolop-provider-example"
    assert catalog.manifest[1].version == "4.5.6"
    assert catalog.manifest[1].identity == f"{resolver.__module__}:{resolver.__qualname__}"

    with raises(FrozenInstanceError):
        setattr(catalog, "capabilities", ())
