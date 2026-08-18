from __future__ import annotations

import base64
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urljoin

from yolop_http import SafeHttpClient

from .registry import OpenAPIOperationError, OpenAPIServiceConfig

SecretResolver = Callable[[str], str | Awaitable[str]]
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class OpenAPIApprovalRequired(OpenAPIOperationError):
    """A write operation requires explicit host approval."""


class OpenAPISecretError(OpenAPIOperationError):
    """A host secret could not be resolved."""


@dataclass
class OpenAPICaller:
    """Execute one host-authorized OpenAPI operation through SafeHttpClient."""

    service: OpenAPIServiceConfig
    client: SafeHttpClient
    secret_resolver: SecretResolver | None = None

    async def call(
        self,
        operation_id: str,
        *,
        path: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        cookies: Mapping[str, str] | None = None,
        body: Any = None,
        approved: bool = False,
    ) -> dict[str, Any]:
        if (
            self.service.allowed_operations is not None
            and operation_id not in self.service.allowed_operations
        ):
            raise OpenAPIOperationError("Operation is outside the host allowlist")
        operation = _operation(self.service.document, operation_id)
        method = operation["method"]
        if method in _WRITE_METHODS and not approved:
            raise OpenAPIApprovalRequired("Write and delete operations require approval")
        path_values = dict(path or {})
        query_values = dict(query or {})
        header_values = dict(headers or {})
        cookie_values = dict(cookies or {})
        path_template = operation["path"]
        parameter_specs = _parameters(self.service.document, path_template, operation_id)
        known = {str(parameter["name"]) for parameter in parameter_specs}
        supplied = set(path_values) | set(query_values) | set(header_values) | set(cookie_values)
        if not supplied.issubset(known):
            raise OpenAPIOperationError("Unknown OpenAPI parameter")
        for parameter in parameter_specs:
            name = str(parameter["name"])
            location = str(parameter["in"])
            values = {
                "path": path_values,
                "query": query_values,
                "header": header_values,
                "cookie": cookie_values,
            }[location]
            if name not in values:
                if parameter.get("required"):
                    raise OpenAPIOperationError(f"Missing required parameter: {name}")
                continue
            value = values[name]
            if location == "path":
                path_template = path_template.replace(
                    "{" + name + "}", quote(str(value), safe="")
                )
            elif location == "query":
                query_values[name] = value
            elif location == "header":
                header_values[name] = str(value)
        resolved_headers = await self._authentication(
            operation,
            header_values,
            query_values,
            cookie_values,
        )
        url = urljoin(self.service.server_url.rstrip("/") + "/", path_template.lstrip("/"))
        if query_values:
            url = f"{url}?{urlencode(query_values, doseq=True)}"
        if cookie_values:
            resolved_headers["Cookie"] = "; ".join(
                f"{name}={value}" for name, value in sorted(cookie_values.items())
            )
        content, content_type = _request_body(operation, body)
        if content_type is not None:
            resolved_headers.setdefault("Content-Type", content_type)
        response = await self.client.request(
            method,
            url,
            headers=resolved_headers,
            content=content,
        )
        try:
            response_content_type = response.headers.get("content-type", "").split(";", 1)[0]
            if response_content_type == "application/json":
                response_body: Any = response.json()
            else:
                response_body = response.text
            return {
                "operation_id": operation_id,
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": response_body,
                "final_url": str(response.url),
                "effect": _effect(operation),
            }
        finally:
            await response.aclose()

    async def _authentication(
        self,
        operation: Mapping[str, Any],
        headers: dict[str, str],
        query: dict[str, Any],
        cookies: dict[str, str],
    ) -> dict[str, str]:
        security = operation.get("security") or self.service.document.get("security") or []
        if not security:
            return headers
        if self.secret_resolver is None:
            raise OpenAPISecretError("OpenAPI secret resolver is required")
        requirement = security[0]
        if not isinstance(requirement, Mapping) or not requirement:
            raise OpenAPISecretError("OpenAPI security requirement is invalid")
        scheme_name = next(iter(requirement))
        scheme = self.service.document.get("components", {}).get("securitySchemes", {}).get(
            scheme_name
        )
        config = self.service.security_schemes.get(scheme_name, {})
        if not isinstance(scheme, Mapping):
            raise OpenAPISecretError("OpenAPI security scheme is missing")
        scheme_type = config.get("type", scheme.get("type"))
        reference = config.get("secret_ref")
        if not isinstance(reference, str):
            raise OpenAPISecretError("OpenAPI secret reference is missing")
        secret = self.secret_resolver(reference)
        if inspect.isawaitable(secret):
            secret = await secret
        if not isinstance(secret, str) or not secret:
            raise OpenAPISecretError("OpenAPI secret resolution failed")
        if scheme_type == "apiKey":
            location = config.get("in", scheme.get("in"))
            name = config.get("name", scheme.get("name"))
            if location == "header":
                headers[str(name)] = secret
            elif location == "query":
                query[str(name)] = secret
            elif location == "cookie":
                cookies[str(name)] = secret
            else:
                raise OpenAPISecretError("OpenAPI API key location is invalid")
        elif scheme_type == "basic":
            username = str(config.get("username", ""))
            headers["Authorization"] = "Basic " + base64.b64encode(
                f"{username}:{secret}".encode()
            ).decode()
        else:
            headers["Authorization"] = f"Bearer {secret}"
        return headers


def _operation(document: Mapping[str, Any], operation_id: str) -> dict[str, Any]:
    for path, path_item in document.get("paths", {}).items():
        if not isinstance(path_item, Mapping):
            continue
        for method, operation in path_item.items():
            if isinstance(operation, Mapping) and operation.get("operationId") == operation_id:
                return {"path": path, "method": method.upper(), **operation}
    raise OpenAPIOperationError(f"Unknown OpenAPI operation: {operation_id}")


def _parameters(
    document: Mapping[str, Any],
    path: str,
    operation_id: str,
) -> list[Mapping[str, Any]]:
    operation = _operation(document, operation_id)
    path_item = document.get("paths", {}).get(path, {})
    values = []
    if isinstance(path_item, Mapping) and isinstance(path_item.get("parameters"), list):
        values.extend(value for value in path_item["parameters"] if isinstance(value, Mapping))
    if isinstance(operation.get("parameters"), list):
        values.extend(value for value in operation["parameters"] if isinstance(value, Mapping))
    return values


def _request_body(
    operation: Mapping[str, Any],
    body: Any,
) -> tuple[bytes | str | None, str | None]:
    if body is None:
        return None, None
    content = operation.get("requestBody", {}).get("content", {})
    if "application/json" in content:
        return json.dumps(body, separators=(",", ":")), "application/json"
    if "application/x-www-form-urlencoded" in content:
        return urlencode(body, doseq=True), "application/x-www-form-urlencoded"
    if "text/plain" in content:
        return str(body), "text/plain"
    raise OpenAPIOperationError("Request body content type is not supported")


def _effect(operation: Mapping[str, Any]) -> str:
    default = "read" if operation["method"] == "GET" else "write"
    return str(operation.get("x-yolop-effect", default))


__all__ = ["OpenAPIApprovalRequired", "OpenAPICaller", "OpenAPISecretError", "SecretResolver"]
