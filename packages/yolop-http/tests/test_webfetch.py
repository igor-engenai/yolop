from types import SimpleNamespace
from typing import Any, cast

import httpx
from pydantic_ai import ModelRetry
from pytest import raises
from yolop_http import EgressPolicy, SafeHttpClient, WebFetch


class FakeClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response

    async def get(self, url: str, *, headers: dict[str, str] | None = None) -> httpx.Response:
        del url, headers
        return self.response


async def _run_fetch(
    response: httpx.Response,
    *,
    max_content_bytes: int = 1_000,
    max_text_chars: int = 1_000,
):
    capability = WebFetch(
        max_content_bytes=max_content_bytes,
        max_text_chars=max_text_chars,
    )
    return await capability.for_run(
        cast(
            Any,
            SimpleNamespace(deps=SimpleNamespace(http_client=FakeClient(response))),
        )
    )


def test_webfetch_capability_is_public() -> None:
    assert WebFetch.__name__ == "WebFetch"


async def test_webfetch_extracts_bounded_html_text() -> None:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(
        200,
        headers={"content-type": "text/html"},
        content=b"<title> A page </title><script>secret()</script><p>Hello</p>",
        request=request,
    )
    run = await _run_fetch(response)

    result = await run.fetch_web("https://example.com")

    assert result["title"] == "A page"
    assert result["content"] == "Hello"
    assert result["status"] == 200


async def test_webfetch_rejects_binary_content_and_bounds_text() -> None:
    binary = httpx.Response(
        200,
        headers={"content-type": "application/octet-stream"},
        content=b"binary",
        request=httpx.Request("GET", "https://example.com"),
    )
    with raises(ModelRetry, match="unsupported"):
        await (await _run_fetch(binary)).fetch_web("https://example.com")

    text = httpx.Response(
        200,
        headers={"content-type": "text/plain"},
        content=b"abcdef",
        request=httpx.Request("GET", "https://example.com"),
    )
    result = await (await _run_fetch(text, max_text_chars=4)).fetch_web(
        "https://example.com"
    )
    assert result["truncated"] is True
    assert result["content"] == "abcd"


async def test_webfetch_cannot_broaden_host_policy() -> None:
    client = SafeHttpClient(
        EgressPolicy(allowed_hosts=frozenset({"example.com"})),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                content=b"ok",
                request=request,
            )
        ),
    )
    try:
        with raises(ValueError, match="host"):
            await client.get("https://evil.example")
    finally:
        await client.aclose()
