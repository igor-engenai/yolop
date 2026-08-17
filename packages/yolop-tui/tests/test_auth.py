from importlib import metadata
from types import SimpleNamespace

from pytest import MonkeyPatch, raises
from yolop_tui.auth import load_auth_providers


class _AuthProvider:
    name = "example"
    label = "Example subscription"

    async def login(self, _notify):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError

    def logout(self) -> bool:
        return False


def test_tui_discovers_installed_auth_providers(monkeypatch: MonkeyPatch) -> None:
    entry_point = SimpleNamespace(name="example", load=lambda: _AuthProvider)
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: (entry_point,))

    providers = load_auth_providers()

    assert len(providers) == 1
    assert providers[0].name == "example"
    assert providers[0].label == "Example subscription"


def test_tui_auth_provider_discovery_allows_no_installed_providers(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: ())

    assert load_auth_providers() == ()


def test_tui_auth_provider_discovery_rejects_duplicate_factories(
    monkeypatch: MonkeyPatch,
) -> None:
    entry_points = (
        SimpleNamespace(name="example", load=lambda: _AuthProvider),
        SimpleNamespace(name="example", load=lambda: _AuthProvider),
    )
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: entry_points)

    with raises(ValueError, match="provider 'example' has multiple installed factories"):
        load_auth_providers()


def test_tui_auth_provider_discovery_validates_provider_contract(
    monkeypatch: MonkeyPatch,
) -> None:
    entry_point = SimpleNamespace(name="example", load=lambda: lambda: object())
    monkeypatch.setattr(metadata, "entry_points", lambda **_kwargs: (entry_point,))

    with raises(ValueError, match="provider with a different name"):
        load_auth_providers()
