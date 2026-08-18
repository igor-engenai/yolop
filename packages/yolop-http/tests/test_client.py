from __future__ import annotations

import asyncio
import gzip
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


async def test_client_stops_streaming_after_decoded_response_limit() -> None:
    class TrackingStream(httpx.AsyncByteStream):
        consumed_past_limit = False

        async def __aiter__(self):
            yield gzip.compress(b"x" * 10_000)
            self.consumed_past_limit = True
            yield b""

    stream = TrackingStream()
    client = SafeHttpClient(
        EgressPolicy(max_response_bytes=1_000),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-encoding": "gzip"},
                stream=stream,
                request=request,
            )
        ),
    )
    try:
        with raises(ValueError, match="byte limit"):
            await client.get("https://example.com")
    finally:
        await client.aclose()

    assert stream.consumed_past_limit is False


async def test_client_enforces_total_timeout() -> None:
    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.05)
            yield b"ok"

    client = SafeHttpClient(
        EgressPolicy(read_timeout=1, total_timeout=0.01),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=SlowStream(), request=request)
        ),
    )
    try:
        with raises(httpx.TimeoutException, match="total timeout"):
            await client.get("https://example.com")
    finally:
        await client.aclose()


async def test_client_rejects_cross_origin_redirects() -> None:
    requested_hosts: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.host == "example.com":
            return httpx.Response(
                302,
                headers={"location": "https://attacker.example/collect"},
                request=request,
            )
        return httpx.Response(200, content=b"leaked", request=request)

    client = SafeHttpClient(
        EgressPolicy(allow_redirects=True),
        transport=httpx.MockTransport(respond),
    )
    try:
        with raises(ValueError, match="origin"):
            await client.get(
                "https://example.com/start",
                headers={"Authorization": "Bearer secret"},
            )
    finally:
        await client.aclose()

    assert requested_hosts == ["example.com"]


async def test_client_applies_see_other_method_rules() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/write":
            return httpx.Response(
                303,
                headers={"location": "/result"},
                request=request,
            )
        return httpx.Response(200, content=b"done", request=request)

    client = SafeHttpClient(
        EgressPolicy(allow_redirects=True),
        transport=httpx.MockTransport(respond),
    )
    try:
        response = await client.request(
            "POST",
            "https://example.com/write",
            headers={"Content-Type": "application/json"},
            content='{"write":true}',
        )
    finally:
        await client.aclose()

    assert response.text == "done"
    assert [request.method for request in requests] == ["POST", "GET"]
    assert requests[1].content == b""
    assert "content-type" not in requests[1].headers


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
