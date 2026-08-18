from __future__ import annotations

import asyncio
import socket
import ssl
from collections.abc import Awaitable, Callable, Mapping, Sequence
from ipaddress import ip_address
from typing import Any, cast
from urllib.parse import urljoin

import httpcore
import httpx

from . import EgressPolicy, IPAddress, validate_destination, validate_url

Resolver = Callable[[str, int], Awaitable[Sequence[IPAddress]]]


async def resolve_addresses(host: str, port: int) -> Sequence[IPAddress]:
    """Resolve a hostname without performing an unvalidated connection."""
    entries = await asyncio.to_thread(
        socket.getaddrinfo,
        host,
        port,
        type=socket.SOCK_STREAM,
    )
    addresses: list[IPAddress] = []
    for entry in entries:
        address = ip_address(entry[4][0])
        if address not in addresses:
            addresses.append(address)
    return addresses


class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolve and validate each connection target before opening its socket."""

    def __init__(self, policy: EgressPolicy, resolver: Resolver = resolve_addresses) -> None:
        self._policy = policy
        self._resolver = resolver
        self._backend = cast(httpcore.AsyncNetworkBackend, httpcore.AnyIOBackend())

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any | None = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = await self._resolver(host, port)
        safe_addresses = [
            address for address in addresses if validate_destination(address, self._policy)
        ]
        if not safe_addresses:
            raise ValueError("DNS resolved only to forbidden addresses")
        return await self._backend.connect_tcp(
            str(safe_addresses[0]),
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise ValueError("Unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class PinnedTransport(httpx.AsyncBaseTransport):
    """HTTPX transport using a validated address while retaining TLS SNI hostname."""

    def __init__(self, policy: EgressPolicy, resolver: Resolver = resolve_addresses) -> None:
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(),
            network_backend=PinnedNetworkBackend(policy, resolver),
            max_connections=10,
            max_keepalive_connections=10,
        )
        self._max_response_bytes = policy.max_response_bytes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        response = await self._pool.handle_async_request(core_request)
        chunks: list[bytes] = []
        size = 0
        try:
            async for chunk in response.aiter_stream():
                size += len(chunk)
                if size > self._max_response_bytes:
                    raise ValueError("HTTP response exceeds the configured byte limit")
                chunks.append(chunk)
        finally:
            await response.aclose()
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            content=b"".join(chunks),
            request=request,
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


class SafeHttpClient:
    """Host-owned bounded HTTP client with validated DNS connection targets."""

    def __init__(
        self,
        policy: EgressPolicy,
        *,
        resolver: Resolver = resolve_addresses,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._policy = policy
        self._client = httpx.AsyncClient(
            transport=transport or PinnedTransport(policy, resolver),
            follow_redirects=False,
            timeout=httpx.Timeout(
                connect=policy.connect_timeout,
                read=policy.read_timeout,
                write=policy.total_timeout,
                pool=policy.connect_timeout,
            ),
            trust_env=False,
        )

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        content: bytes | str | None = None,
        _redirects: int = 0,
    ) -> httpx.Response:
        parsed = validate_url(url, self._policy)
        hostname = parsed.hostname
        assert hostname is not None
        if (
            self._policy.allowed_hosts is not None
            and hostname.lower() not in self._policy.allowed_hosts
        ):
            raise ValueError("URL host is not allowed")
        response = await self._client.request(method, url, headers=headers, content=content)
        if response.is_redirect:
            location = response.headers.get("location")
            await response.aclose()
            if not self._policy.allow_redirects:
                raise ValueError("HTTP redirects are not allowed")
            if location is None or _redirects >= 3:
                raise ValueError("HTTP redirect is invalid or exceeds the limit")
            redirected_url = urljoin(url, location)
            validate_url(redirected_url, self._policy)
            return await self.request(
                method,
                redirected_url,
                headers=headers,
                content=content,
                _redirects=_redirects + 1,
            )
        if len(response.content) > self._policy.max_response_bytes:
            await response.aclose()
            raise ValueError("HTTP response exceeds the configured byte limit")
        return response

    async def get(self, url: str, *, headers: Mapping[str, str] | None = None) -> httpx.Response:
        return await self.request("GET", url, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> SafeHttpClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


__all__ = [
    "PinnedNetworkBackend",
    "PinnedTransport",
    "Resolver",
    "SafeHttpClient",
    "resolve_addresses",
]
