from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import IPv4Address, IPv6Address
from typing import Final
from urllib.parse import SplitResult, urlsplit

IPAddress = IPv4Address | IPv6Address
_FORBIDDEN_SCHEMES: Final = frozenset({"file", "ftp", "gopher"})


@dataclass(frozen=True)
class EgressPolicy:
    """Immutable host-owned outbound HTTP policy."""

    allowed_schemes: frozenset[str] = field(default_factory=lambda: frozenset({"https"}))
    allowed_ports: frozenset[int] = field(default_factory=lambda: frozenset({443}))
    allowed_hosts: frozenset[str] | None = None
    max_response_bytes: int = 1_048_576
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    total_timeout: float = 60.0
    allow_redirects: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_schemes or any(
            scheme in _FORBIDDEN_SCHEMES for scheme in self.allowed_schemes
        ):
            raise ValueError("Egress policy must allow a safe scheme")
        if not self.allowed_ports or any(port < 1 or port > 65535 for port in self.allowed_ports):
            raise ValueError("Egress policy ports must be valid")
        if self.allowed_hosts is not None and any(not host.strip() for host in self.allowed_hosts):
            raise ValueError("Egress policy hosts must not be empty")
        if self.max_response_bytes < 1:
            raise ValueError("Egress response limit must be positive")
        if min(self.connect_timeout, self.read_timeout, self.total_timeout) <= 0:
            raise ValueError("Egress timeouts must be positive")


def validate_destination(address: IPAddress, policy: EgressPolicy) -> bool:
    """Return whether a resolved address is safe for outbound access."""
    return not (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    )


def validate_url(url: str, policy: EgressPolicy) -> SplitResult:
    """Validate scheme, host, and port before DNS resolution."""
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in policy.allowed_schemes:
        raise ValueError("URL scheme is not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are not allowed")
    if not parsed.hostname:
        raise ValueError("URL host is required")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as error:
        raise ValueError("URL port is invalid") from error
    if port not in policy.allowed_ports:
        raise ValueError("URL port is not allowed")
    return parsed


from .client import (  # noqa: E402
    PinnedNetworkBackend,
    PinnedTransport,
    Resolver,
    SafeHttpClient,
    resolve_addresses,
)
from .webfetch import WebFetch, WebFetchClient  # noqa: E402

__all__ = [
    "EgressPolicy",
    "IPAddress",
    "PinnedNetworkBackend",
    "PinnedTransport",
    "Resolver",
    "SafeHttpClient",
    "resolve_addresses",
    "validate_destination",
    "validate_url",
    "WebFetch",
    "WebFetchClient",
]
