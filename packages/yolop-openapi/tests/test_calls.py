from __future__ import annotations

from typing import Any

import httpx
from pytest import raises
from yolop_http import EgressPolicy, SafeHttpClient
from yolop_openapi import (
    OpenAPIApprovalRequired,
    OpenAPICaller,
    OpenAPIServiceConfig,
)


def document() -> dict[str, Any]:
    return {
        "openapi": "3.0.3",
        "info": {"title": "Example", "version": "1"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users/{id}": {
                "get": {
                    "operationId": "get_user",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True},
                        {"name": "q", "in": "query"},
                        {"name": "X-Trace", "in": "header"},
                        {"name": "sid", "in": "cookie"},
                    ],
                    "responses": {"200": {"description": "ok"}},
                },
            },
            "/users": {
                "post": {
                    "operationId": "create_user",
                    "requestBody": {
                        "content": {"application/json": {"schema": {"type": "object"}}}
                    },
                    "responses": {"201": {"description": "created"}},
                }
            },
        },
    }


async def test_openapi_caller_is_public() -> None:
    assert OpenAPICaller.__name__ == "OpenAPICaller"


async def test_openapi_call_builds_bounded_request_parameters() -> None:
    captured: dict[str, Any] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"id": "42"},
            request=request,
        )

    client = SafeHttpClient(
        EgressPolicy(allowed_hosts=frozenset({"api.example.com"})),
        transport=httpx.MockTransport(respond),
    )
    service = OpenAPIServiceConfig(
        alias="crm",
        document=document(),
        server_url="https://api.example.com",
        allowed_operations=frozenset({"get_user"}),
    )
    try:
        result = await OpenAPICaller(service, client).call(
            "get_user",
            path={"id": "42"},
            query={"q": "active"},
            headers={"X-Trace": "trace"},
            cookies={"sid": "cookie"},
        )
    finally:
        await client.aclose()

    assert captured["url"] == "https://api.example.com/users/42?q=active"
    assert captured["headers"]["x-trace"] == "trace"
    assert captured["headers"]["cookie"] == "sid=cookie"
    assert result["body"] == {"id": "42"}


async def test_openapi_write_requires_approval_and_resolves_api_key() -> None:
    captured: dict[str, Any] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = request.read()
        return httpx.Response(201, json={"created": True}, request=request)

    client = SafeHttpClient(
        EgressPolicy(allowed_hosts=frozenset({"api.example.com"})),
        transport=httpx.MockTransport(respond),
    )
    write_document = document()
    write_document["components"] = {
        "securitySchemes": {
            "apiKey": {"type": "apiKey", "in": "header", "name": "X-Key"}
        }
    }
    write_document["paths"]["/users"]["post"]["security"] = [{"apiKey": []}]
    service = OpenAPIServiceConfig(
        alias="crm",
        document=write_document,
        server_url="https://api.example.com",
        allowed_operations=frozenset({"create_user"}),
        security_schemes={"apiKey": {"secret_ref": "kv/api-key"}},
    )
    caller = OpenAPICaller(service, client, secret_resolver=lambda ref: f"secret:{ref}")
    try:
        with raises(OpenAPIApprovalRequired):
            await caller.call("create_user", body={"name": "Ada"})
        result = await caller.call("create_user", body={"name": "Ada"}, approved=True)
    finally:
        await client.aclose()

    assert captured["headers"]["x-key"] == "secret:kv/api-key"
    assert captured["body"] == b'{"name":"Ada"}'
    assert result["status"] == 201
