from __future__ import annotations

from typing import Any

import httpx
from pytest import raises
from yolop_http import EgressPolicy, SafeHttpClient
from yolop_openapi import (
    OpenAPIApprovalRequired,
    OpenAPICaller,
    OpenAPIOperationError,
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
            cookies={"sid": "x; admin=true"},
        )
    finally:
        await client.aclose()

    assert captured["url"] == "https://api.example.com/users/42?q=active"
    assert captured["headers"]["x-trace"] == "trace"
    assert captured["headers"]["cookie"] == 'sid="x\\073 admin=true"'
    assert result["body"] == {"id": "42"}


async def test_openapi_call_rejects_parameter_in_wrong_location() -> None:
    client = SafeHttpClient(
        EgressPolicy(allowed_hosts=frozenset({"api.example.com"})),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"ok", request=request)
        ),
    )
    service = OpenAPIServiceConfig(
        alias="crm",
        document=document(),
        server_url="https://api.example.com",
        allowed_operations=frozenset({"get_user"}),
    )
    try:
        with raises(OpenAPIOperationError, match="location"):
            await OpenAPICaller(service, client).call(
                "get_user",
                path={"id": "42"},
                headers={"q": "smuggled"},
            )
    finally:
        await client.aclose()


async def test_openapi_operation_can_disable_global_security() -> None:
    captured: dict[str, Any] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, content=b"ok", request=request)

    public_document = document()
    public_document["components"] = {
        "securitySchemes": {"apiKey": {"type": "apiKey", "in": "header", "name": "X-Key"}}
    }
    public_document["security"] = [{"apiKey": []}]
    public_document["paths"]["/users/{id}"]["get"]["security"] = []
    client = SafeHttpClient(
        EgressPolicy(allowed_hosts=frozenset({"api.example.com"})),
        transport=httpx.MockTransport(respond),
    )
    service = OpenAPIServiceConfig(
        alias="crm",
        document=public_document,
        server_url="https://api.example.com",
        allowed_operations=frozenset({"get_user"}),
        security_schemes={"apiKey": {"secret_ref": "kv/api-key"}},
    )
    try:
        await OpenAPICaller(service, client).call("get_user", path={"id": "42"})
    finally:
        await client.aclose()

    assert "x-key" not in captured["headers"]


async def test_openapi_security_requirement_applies_all_schemes() -> None:
    captured: dict[str, Any] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["query"] = dict(request.url.params)
        return httpx.Response(201, content=b"ok", request=request)

    secured_document = document()
    secured_document["components"] = {
        "securitySchemes": {
            "headerKey": {"type": "apiKey", "in": "header", "name": "X-Key"},
            "queryKey": {"type": "apiKey", "in": "query", "name": "api_key"},
        }
    }
    secured_document["paths"]["/users"]["post"]["security"] = [{"headerKey": [], "queryKey": []}]
    client = SafeHttpClient(
        EgressPolicy(allowed_hosts=frozenset({"api.example.com"})),
        transport=httpx.MockTransport(respond),
    )
    service = OpenAPIServiceConfig(
        alias="crm",
        document=secured_document,
        server_url="https://api.example.com",
        allowed_operations=frozenset({"create_user"}),
        security_schemes={
            "headerKey": {"secret_ref": "kv/header"},
            "queryKey": {"secret_ref": "kv/query"},
        },
    )
    caller = OpenAPICaller(service, client, secret_resolver=lambda ref: f"secret:{ref}")
    try:
        await caller.call("create_user", body={"name": "Ada"}, approved=True)
    finally:
        await client.aclose()

    assert captured["headers"]["x-key"] == "secret:kv/header"
    assert captured["query"]["api_key"] == "secret:kv/query"


async def test_openapi_standard_http_basic_authentication() -> None:
    captured: dict[str, Any] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, content=b"ok", request=request)

    basic_document = document()
    basic_document["components"] = {
        "securitySchemes": {
            "basicAuth": {"type": "http", "scheme": "basic"},
        }
    }
    basic_document["paths"]["/users/{id}"]["get"]["security"] = [{"basicAuth": []}]
    client = SafeHttpClient(
        EgressPolicy(allowed_hosts=frozenset({"api.example.com"})),
        transport=httpx.MockTransport(respond),
    )
    service = OpenAPIServiceConfig(
        alias="crm",
        document=basic_document,
        server_url="https://api.example.com",
        allowed_operations=frozenset({"get_user"}),
        security_schemes={"basicAuth": {"secret_ref": "kv/password", "username": "ada"}},
    )
    caller = OpenAPICaller(service, client, secret_resolver=lambda _ref: "password")
    try:
        await caller.call("get_user", path={"id": "42"})
    finally:
        await client.aclose()

    assert captured["headers"]["authorization"] == "Basic YWRhOnBhc3N3b3Jk"


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
        "securitySchemes": {"apiKey": {"type": "apiKey", "in": "header", "name": "X-Key"}}
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
