from __future__ import annotations

from ipaddress import IPv4Address
from typing import Any, cast

import httpx
from pytest import raises
from yolop_http import EgressPolicy, PinnedNetworkBackend, SafeHttpClient, validate_url


async def test_dns_rejects_forbidden_targets() -> None:
    async def resolver(_host: str, _port: int):
        return [IPv4Address("169.254.169.254")]

    backend = PinnedNetworkBackend(EgressPolicy(), resolver)

    with raises(ValueError, match="forbidden"):
        await backend.connect_tcp("metadata.internal", 443)


async def test_dns_rebinding_uses_the_validated_public_address() -> None:
    async def resolver(_host: str, _port: int):
        return [IPv4Address("8.8.8.8")]

    backend = PinnedNetworkBackend(EgressPolicy(), resolver)
    selected: list[str] = []

    async def connect_tcp(host: str, *_args: Any, **_kwargs: Any) -> object:
        selected.append(host)
        return object()

    cast(Any, backend._backend).connect_tcp = connect_tcp
    await backend.connect_tcp("public.example", 443)

    assert selected == ["8.8.8.8"]


async def test_url_rejects_credentials_and_forbidden_port() -> None:
    policy = EgressPolicy()

    with raises(ValueError):
        validate_url("https://user:password@example.com", policy)
    with raises(ValueError):
        validate_url("https://example.com:8443", policy)


async def test_client_rejects_redirect_and_oversized_response() -> None:
    redirect_client = SafeHttpClient(
        EgressPolicy(),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                302,
                headers={"location": "https://example.com/next"},
            )
        ),
    )
    try:
        with raises(ValueError, match="redirect"):
            await redirect_client.get("https://example.com")
    finally:
        await redirect_client.aclose()

    oversized_client = SafeHttpClient(
        EgressPolicy(max_response_bytes=3),
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"1234")),
    )
    try:
        with raises(ValueError, match="byte limit"):
            await oversized_client.get("https://example.com")
    finally:
        await oversized_client.aclose()
