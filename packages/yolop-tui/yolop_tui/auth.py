from collections.abc import Callable
from importlib import metadata
from typing import Any, Protocol, cast

_ENTRY_POINT_GROUP = "yolop.auth_providers"


class DeviceAuthorization(Protocol):
    verification_uri: str
    user_code: str
    expires_in: float


class AuthStatus(Protocol):
    authenticated: bool
    expires_at: float | None


class AuthProvider(Protocol):
    name: str
    label: str

    async def login(self, notify: Callable[[DeviceAuthorization], None]) -> AuthStatus: ...

    def status(self) -> AuthStatus: ...

    def logout(self) -> bool: ...


def load_auth_providers() -> tuple[AuthProvider, ...]:
    """Load installed authentication providers for local TUI commands."""
    entry_points = metadata.entry_points(group=_ENTRY_POINT_GROUP)
    providers: list[AuthProvider] = []
    seen: set[str] = set()
    for entry_point in sorted(entry_points, key=lambda item: item.name):
        if entry_point.name in seen:
            raise ValueError(
                f"Authentication provider {entry_point.name!r} has multiple installed factories"
            )
        seen.add(entry_point.name)
        factory = entry_point.load()
        if not callable(factory):
            raise TypeError(
                f"Authentication provider {entry_point.name!r} did not load a callable factory"
            )
        provider = factory()
        _validate_provider(entry_point.name, provider)
        providers.append(cast(AuthProvider, provider))
    return tuple(providers)


def _validate_provider(entry_point_name: str, provider: Any) -> None:
    if getattr(provider, "name", None) != entry_point_name:
        raise ValueError(
            f"Authentication provider {entry_point_name!r} loaded a provider with a different name"
        )
    if not isinstance(getattr(provider, "label", None), str):
        raise TypeError(f"Authentication provider {entry_point_name!r} has no label")
    for method_name in ("login", "status", "logout"):
        if not callable(getattr(provider, method_name, None)):
            raise TypeError(
                f"Authentication provider {entry_point_name!r} has no {method_name} method"
            )
