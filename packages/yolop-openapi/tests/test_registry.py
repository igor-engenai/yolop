from __future__ import annotations

from copy import deepcopy
from typing import Any

from pytest import raises
from yolop_openapi import (
    OpenAPIConfigurationError,
    OpenAPIExplorer,
    OpenAPIForbiddenAliasError,
    OpenAPIOperationError,
    OpenAPIRegistry,
    OpenAPIServerError,
    OpenAPIServiceConfig,
    OpenAPIUnknownAliasError,
)

DOCUMENT: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Example", "version": "1"},
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/users": {
            "get": {
                "operationId": "list_users",
                "responses": {"200": {"description": "ok"}},
            },
        },
        "/users/{id}": {
            "delete": {
                "operationId": "delete_user",
                "responses": {"204": {"description": "deleted"}},
            },
        },
    },
}


def service(alias: str = "crm", **kwargs: Any) -> OpenAPIServiceConfig:
    return OpenAPIServiceConfig(
        alias=alias,
        document=DOCUMENT,
        server_url="https://api.example.com",
        **kwargs,
    )


def test_openapi_registry_is_public() -> None:
    assert OpenAPIRegistry.__name__ == "OpenAPIRegistry"


def test_registry_rejects_invalid_document_and_server() -> None:
    with raises(OpenAPIConfigurationError):
        OpenAPIServiceConfig(
            alias="bad",
            document={"openapi": "2.0", "servers": [], "paths": {}},
            server_url="https://api.example.com",
        )
    with raises(OpenAPIServerError):
        OpenAPIServiceConfig(
            alias="bad",
            document=DOCUMENT,
            server_url="https://other.example.com",
        )
    with raises(OpenAPIOperationError):
        OpenAPIServiceConfig(
            alias="bad",
            document={
                **DOCUMENT,
                "paths": {
                    "/users": {
                        "get": {"responses": {"200": {"description": "missing id"}}}
                    }
                },
            },
            server_url="https://api.example.com",
        )


def test_service_config_deeply_freezes_the_pinned_document() -> None:
    document = deepcopy(DOCUMENT)
    configured = OpenAPIServiceConfig(
        alias="crm",
        document=document,
        server_url="https://api.example.com",
    )
    digest = configured.spec_digest

    document["paths"]["/users"]["get"]["operationId"] = "mutated"

    assert configured.spec_digest == digest
    assert set(configured.operations) == {"list_users", "delete_user"}
    with raises(TypeError):
        configured.document["paths"]["/users"]["get"]["operationId"] = "blocked"


def test_registry_rejects_duplicate_alias_and_prefix() -> None:
    with raises(OpenAPIConfigurationError):
        OpenAPIRegistry([service(), service()])
    with raises(OpenAPIConfigurationError):
        OpenAPIRegistry([service(), service(alias="other", tool_prefix="crm")])


def test_registry_rejects_unknown_and_forbidden_aliases() -> None:
    registry = OpenAPIRegistry([service()], allowed_aliases=frozenset())
    with raises(OpenAPIUnknownAliasError):
        registry.build_for_spec({"metadata": {"openapi": ["unknown"]}})
    with raises(OpenAPIForbiddenAliasError):
        OpenAPIRegistry([service()], allowed_aliases=frozenset()).build_for_spec(
            {"metadata": {"openapi": ["crm"]}}
        )


def test_registry_intersects_agent_selection_with_host_allowlist_and_is_stable() -> None:
    registry = OpenAPIRegistry([service(allowed_operations=frozenset({"list_users"}))])
    composition = registry.build_for_spec(
        {
            "metadata": {
                "openapi": [{"alias": "crm", "operations": ["list_users"]}]
            }
        }
    )
    explorer = composition.capabilities[0]
    assert isinstance(explorer, OpenAPIExplorer)

    assert explorer.catalog == (
        {"operation_id": "list_users", "method": "GET", "path": "/users"},
    )

    with raises(OpenAPIOperationError):
        registry.build_for_spec(
            {"metadata": {"openapi": [{"alias": "crm", "operations": ["delete_user"]}]}}
        )
