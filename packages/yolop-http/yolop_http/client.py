from __future__ import annotations

import asyncio
import socket
import ssl
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from ipaddress import ip_address
from typing import Any, cast
from urllib.parse import urljoin

import httpcore
import httpx

from . import EgressPolicy, IPAddress, validate_destination, validate_url

Resolver = Callable[[str, int], Awaitable[Sequence[IPAddress]]]


def _origin(url: httpx.URL) -> tuple[str, str, int]:
    port = url.port or (443 if url.scheme == "https" else 80)
    return url.scheme, url.host.lower(), port


def _redirect_request(
    status_code: int,
    method: str,
    headers: Mapping[str, str] | None,
    content: bytes | str | None,
) -> tuple[str, Mapping[str, str] | None, bytes | str | None]:
    normalized_method = method.upper()
    switch_to_get = (
        (status_code == 303 and normalized_method != "HEAD")
        or status_code in {301, 302}
        and normalized_method == "POST"
    )
    if not switch_to_get:
        return method, headers, content
    body_headers = {"content-length", "content-type", "transfer-encoding"}
    redirected_headers = (
        None
        if headers is None
        else {name: value for name, value in headers.items() if name.lower() not in body_headers}
    )
    return "GET", redirected_headers, None


async def _read_bounded_response(response: httpx.Response, limit: int) -> bytes:
    if response.is_stream_consumed:
        if len(response.content) > limit:
            raise ValueError("HTTP response exceeds the configured byte limit")
        return response.content
    encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if encoding not in {"", "identity", "gzip"}:
        raise ValueError("HTTP response uses an unsupported content encoding")
    chunks: list[bytes] = []
    remaining = limit

    def append(chunk: bytes) -> None:
        nonlocal remaining
        if len(chunk) > remaining:
            raise ValueError("HTTP response exceeds the configured byte limit")
        if chunk:
            chunks.append(chunk)
            remaining -= len(chunk)

    if encoding in {"", "identity"}:
        async for chunk in response.aiter_raw():
            append(chunk)
        return b"".join(chunks)

    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    async for raw_chunk in response.aiter_raw():
        pending = raw_chunk
        while pending:
            previous_size = len(pending)
            append(decoder.decompress(pending, remaining + 1))
            if decoder.unused_data:
                raise ValueError("HTTP response contains concatenated gzip streams")
            pending = decoder.unconsumed_tail
            if pending and len(pending) == previous_size:
                raise ValueError("HTTP response gzip stream made no progress")
    if not decoder.eof:
        raise ValueError("HTTP response contains an incomplete gzip stream")
    return b"".join(chunks)


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


class _CoreResponseStream(httpx.AsyncByteStream):
    def __init__(self, response: httpcore.Response) -> None:
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_stream():
            yield chunk

    async def aclose(self) -> None:
        await self._response.aclose()


class PinnedTransport(httpx.AsyncBaseTransport):
    """HTTPX transport using a validated address while retaining TLS SNI hostname."""

    def __init__(self, policy: EgressPolicy, resolver: Resolver = resolve_addresses) -> None:
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(),
            network_backend=PinnedNetworkBackend(policy, resolver),
            max_connections=10,
            max_keepalive_connections=10,
        )

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
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_CoreResponseStream(response),
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
            headers={"Accept-Encoding": "gzip"},
            follow_redirects=False,
            timeout=httpx.Timeout(
                connect=policy.connect_timeout,
                read=policy.read_timeout,
                write=policy.read_timeout,
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
    ) -> httpx.Response:
        try:
            async with asyncio.timeout(self._policy.total_timeout):
                return await self._request(
                    method,
                    url,
                    headers=headers,
                    content=content,
                )
        except TimeoutError as error:
            raise httpx.TimeoutException("HTTP request exceeded the total timeout") from error

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        content: bytes | str | None,
        redirects: int = 0,
    ) -> httpx.Response:
        parsed = validate_url(url, self._policy)
        hostname = parsed.hostname
        assert hostname is not None
        if (
            self._policy.allowed_hosts is not None
            and hostname.lower() not in self._policy.allowed_hosts
        ):
            raise ValueError("URL host is not allowed")
        response = await self._send_bounded(method, url, headers=headers, content=content)
        if response.is_redirect:
            location = response.headers.get("location")
            await response.aclose()
            if not self._policy.allow_redirects:
                raise ValueError("HTTP redirects are not allowed")
            if location is None or redirects >= 3:
                raise ValueError("HTTP redirect is invalid or exceeds the limit")
            redirected_url = urljoin(url, location)
            validate_url(redirected_url, self._policy)
            if _origin(httpx.URL(redirected_url)) != _origin(httpx.URL(url)):
                raise ValueError("HTTP redirect changed origin")
            redirected_method, redirected_headers, redirected_content = _redirect_request(
                response.status_code,
                method,
                headers,
                content,
            )
            return await self._request(
                redirected_method,
                redirected_url,
                headers=redirected_headers,
                content=redirected_content,
                redirects=redirects + 1,
            )
        if len(response.content) > self._policy.max_response_bytes:
            await response.aclose()
            raise ValueError("HTTP response exceeds the configured byte limit")
        return response

    async def _send_bounded(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        content: bytes | str | None,
    ) -> httpx.Response:
        request = self._client.build_request(method, url, headers=headers, content=content)
        response = await self._client.send(request, stream=True)
        if response.is_redirect:
            return response
        try:
            content = await _read_bounded_response(
                response,
                self._policy.max_response_bytes,
            )
        finally:
            await response.aclose()
        response_headers = response.headers.copy()
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)
        return httpx.Response(
            status_code=response.status_code,
            headers=response_headers,
            content=content,
            request=request,
            extensions=response.extensions,
        )

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
