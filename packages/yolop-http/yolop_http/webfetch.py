from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Protocol

import httpx
from pydantic_ai import ModelRetry, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset


class WebFetchClient(Protocol):
    async def get(self, url: str, *, headers: dict[str, str] | None = None) -> httpx.Response: ...


@dataclass
class WebFetch(AbstractCapability[Any]):
    """Fetch bounded public text through a host-owned SafeHttpClient."""

    max_content_bytes: int = 1_048_576
    max_text_chars: int = 100_000

    def __post_init__(self) -> None:
        if self.max_content_bytes < 1 or self.max_text_chars < 1:
            raise ValueError("WebFetch limits must be positive")

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        client = getattr(ctx.deps, "http_client", None)
        if client is None or not hasattr(client, "get"):
            raise ValueError("WebFetch requires deps.http_client")
        return _WebFetchRun(
            client=client,
            max_content_bytes=self.max_content_bytes,
            max_text_chars=self.max_text_chars,
        )


@dataclass
class _WebFetchRun(AbstractCapability[Any]):
    client: WebFetchClient
    max_content_bytes: int
    max_text_chars: int
    _toolset: FunctionToolset[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._toolset = FunctionToolset(
            [Tool(self.fetch_web, takes_ctx=False)],
            id="web-fetch",
        )

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    async def fetch_web(self, url: str) -> dict[str, Any]:
        """Fetch and normalize one bounded public text resource."""
        try:
            response = await self.client.get(url)
        except (httpx.HTTPError, ValueError) as error:
            raise ModelRetry(f"Web fetch was rejected: {error}") from error
        try:
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type not in {
                "text/html",
                "text/plain",
                "application/json",
                "application/xml",
            }:
                raise ModelRetry("Web fetch returned unsupported content")
            if len(response.content) > self.max_content_bytes:
                raise ModelRetry("Web fetch response exceeded the content limit")
            text = response.text
            title = ""
            if content_type == "text/html":
                title, text = _extract_html(text)
            truncated = len(text) > self.max_text_chars
            text = text[: self.max_text_chars]
            return {
                "title": title,
                "content": text,
                "status": response.status_code,
                "final_url": str(response.url),
                "truncated": truncated,
            }
        finally:
            response.close()


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._title_depth = 0
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "title":
            self._title_depth += 1
        elif tag in {"script", "style", "noscript", "template"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        elif tag in {"script", "style", "noscript", "template"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._title_depth:
            self.title_parts.append(data)
        else:
            self.text_parts.append(data)


def _extract_html(content: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(content)
    title = " ".join("".join(parser.title_parts).split())
    text = " ".join(" ".join(parser.text_parts).split())
    return title, text


__all__ = ["WebFetch", "WebFetchClient"]
