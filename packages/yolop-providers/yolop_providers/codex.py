import asyncio
import base64
import json
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from .credentials import CredentialStatus, CredentialStore, OAuthCredential

_PROVIDER_NAME = "openai-codex"
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_AUTH_BASE_URL = "https://auth.openai.com"
_DEVICE_CODE_URL = f"{_AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
_DEVICE_TOKEN_URL = f"{_AUTH_BASE_URL}/api/accounts/deviceauth/token"
_TOKEN_URL = f"{_AUTH_BASE_URL}/oauth/token"
_DEVICE_VERIFICATION_URI = f"{_AUTH_BASE_URL}/codex/device"
_DEVICE_REDIRECT_URI = f"{_AUTH_BASE_URL}/deviceauth/callback"
_DEVICE_TIMEOUT_SECONDS = 15 * 60.0
_REFRESH_SKEW_SECONDS = 60.0
_JWT_CLAIM = "https://api.openai.com/auth"


class CodexOAuthError(RuntimeError):
    """A safe error raised by the OpenAI Codex OAuth flow."""


class CodexNotAuthenticatedError(CodexOAuthError):
    """OpenAI Codex has no stored YoloP credential."""


@dataclass(frozen=True)
class DeviceAuthorization:
    verification_uri: str
    user_code: str
    expires_in: float


@dataclass(frozen=True)
class _DeviceCode:
    device_auth_id: str
    user_code: str
    interval: float


class CodexOAuth:
    """Own OpenAI Codex device login and credentials for one local YoloP user."""

    name = _PROVIDER_NAME
    label = "OpenAI Codex (ChatGPT Plus/Pro)"

    def __init__(
        self,
        *,
        store: CredentialStore | None = None,
        http_client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now: Callable[[], float] = time.time,
        device_timeout: float = _DEVICE_TIMEOUT_SECONDS,
    ) -> None:
        self.store = store or CredentialStore()
        self._http_client = http_client
        self._sleep = sleep
        self._now = now
        self._device_timeout = device_timeout

    async def login(
        self,
        notify: Callable[[DeviceAuthorization], None],
    ) -> CredentialStatus:
        async with self._client() as client:
            device = await self._start_device_login(client)
            notify(
                DeviceAuthorization(
                    verification_uri=_DEVICE_VERIFICATION_URI,
                    user_code=device.user_code,
                    expires_in=self._device_timeout,
                )
            )
            try:
                async with asyncio.timeout(self._device_timeout):
                    authorization_code, code_verifier = await self._poll_device_login(
                        client,
                        device,
                    )
                    credential = await self._exchange_code(
                        client,
                        authorization_code=authorization_code,
                        code_verifier=code_verifier,
                    )
            except TimeoutError as error:
                raise CodexOAuthError("OpenAI Codex device login timed out") from error
        self.store.save_oauth(_PROVIDER_NAME, credential)
        return self.store.status(_PROVIDER_NAME)

    async def access_token(self) -> str:
        """Return a usable token, refreshing and rotating credentials when near expiry."""
        async with self.store.transaction() as transaction:
            credential = transaction.load_oauth(_PROVIDER_NAME)
            if credential is None:
                raise CodexNotAuthenticatedError(
                    "OpenAI Codex is not logged in; use /login or "
                    "`yolop-providers login openai-codex`"
                )
            if credential.expires_at > self._now() + _REFRESH_SKEW_SECONDS:
                return credential.access_token.get_secret_value()
        return await self.refresh_access_token()

    async def refresh_access_token(self, *, stale_access_token: str | None = None) -> str:
        """Force a token refresh after the API rejects the current access token."""
        async with self.store.transaction() as transaction:
            credential = transaction.load_oauth(_PROVIDER_NAME)
            if credential is None:
                raise CodexNotAuthenticatedError(
                    "OpenAI Codex is not logged in; use /login or "
                    "`yolop-providers login openai-codex`"
                )
            current_access_token = credential.access_token.get_secret_value()
            if stale_access_token is not None and current_access_token != stale_access_token:
                return current_access_token
            async with self._client() as client:
                refreshed = await self._refresh(client, credential)
            transaction.save_oauth(_PROVIDER_NAME, refreshed)
            return refreshed.access_token.get_secret_value()

    def status(self) -> CredentialStatus:
        return self.store.status(_PROVIDER_NAME)

    def logout(self) -> bool:
        return self.store.logout(_PROVIDER_NAME)

    async def _start_device_login(self, client: httpx.AsyncClient) -> _DeviceCode:
        response = await _post(
            client,
            _DEVICE_CODE_URL,
            json={"client_id": _CLIENT_ID},
            operation="device code request",
        )
        payload = _response_object(response, "device code request")
        device_auth_id = payload.get("device_auth_id")
        user_code = payload.get("user_code")
        raw_interval = payload.get("interval")
        if isinstance(raw_interval, bool) or not isinstance(raw_interval, str | int | float):
            raise CodexOAuthError("Invalid OpenAI Codex device code response")
        try:
            interval = float(raw_interval)
        except ValueError as error:
            raise CodexOAuthError("Invalid OpenAI Codex device code response") from error
        if (
            not isinstance(device_auth_id, str)
            or not device_auth_id
            or not isinstance(user_code, str)
            or not user_code
            or not math.isfinite(interval)
            or interval < 0
        ):
            raise CodexOAuthError("Invalid OpenAI Codex device code response")
        return _DeviceCode(device_auth_id, user_code, interval)

    async def _poll_device_login(
        self,
        client: httpx.AsyncClient,
        device: _DeviceCode,
    ) -> tuple[str, str]:
        interval = device.interval
        while True:
            await self._sleep(interval)
            try:
                response = await client.post(
                    _DEVICE_TOKEN_URL,
                    json={
                        "device_auth_id": device.device_auth_id,
                        "user_code": device.user_code,
                    },
                )
            except httpx.HTTPError as error:
                raise CodexOAuthError("OpenAI Codex device authorization failed") from error
            if response.is_success:
                payload = _response_object(response, "device authorization")
                authorization_code = payload.get("authorization_code")
                code_verifier = payload.get("code_verifier")
                if (
                    not isinstance(authorization_code, str)
                    or not authorization_code
                    or not isinstance(code_verifier, str)
                    or not code_verifier
                ):
                    raise CodexOAuthError("Invalid OpenAI Codex device authorization response")
                return authorization_code, code_verifier
            error_code = _response_error_code(response)
            if response.status_code in {403, 404} or error_code in {
                "authorization_pending",
                "deviceauth_authorization_pending",
            }:
                continue
            if error_code == "slow_down":
                interval += 5
                continue
            raise CodexOAuthError(
                f"OpenAI Codex device authorization failed ({response.status_code})"
            )

    async def _exchange_code(
        self,
        client: httpx.AsyncClient,
        *,
        authorization_code: str,
        code_verifier: str,
    ) -> OAuthCredential:
        response = await _post(
            client,
            _TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": _CLIENT_ID,
                "code": authorization_code,
                "code_verifier": code_verifier,
                "redirect_uri": _DEVICE_REDIRECT_URI,
            },
            operation="token exchange",
        )
        return _credential_from_token_response(response, now=self._now())

    async def _refresh(
        self,
        client: httpx.AsyncClient,
        credential: OAuthCredential,
    ) -> OAuthCredential:
        response = await _post(
            client,
            _TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": credential.refresh_token.get_secret_value(),
                "client_id": _CLIENT_ID,
            },
            operation="token refresh",
        )
        return _credential_from_token_response(response, now=self._now())

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
            return
        async with httpx.AsyncClient() as client:
            yield client


async def _post(
    client: httpx.AsyncClient,
    url: str,
    *,
    operation: str,
    **kwargs: Any,
) -> httpx.Response:
    try:
        response = await client.post(url, **kwargs)
    except httpx.HTTPError as error:
        raise CodexOAuthError(f"OpenAI Codex {operation} failed") from error
    if not response.is_success:
        raise CodexOAuthError(f"OpenAI Codex {operation} failed ({response.status_code})")
    return response


def _response_object(response: httpx.Response, operation: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeError) as error:
        raise CodexOAuthError(f"Invalid OpenAI Codex {operation} response") from error
    if not isinstance(payload, dict):
        raise CodexOAuthError(f"Invalid OpenAI Codex {operation} response")
    return payload


def _response_error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return error["code"]
    return None


def _credential_from_token_response(response: httpx.Response, *, now: float) -> OAuthCredential:
    payload = _response_object(response, "token")
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(refresh_token, str)
        or not refresh_token
        or isinstance(expires_in, bool)
        or not isinstance(expires_in, int | float)
        or not math.isfinite(expires_in)
        or expires_in <= 0
    ):
        raise CodexOAuthError("Invalid OpenAI Codex token response")
    account_id = _account_id(access_token)
    return OAuthCredential(
        access_token=SecretStr(access_token),
        refresh_token=SecretStr(refresh_token),
        expires_at=now + float(expires_in),
        account_id=account_id,
    )


def _account_id(access_token: str) -> str:
    try:
        parts = access_token.split(".")
        if len(parts) != 3:
            raise ValueError
        encoded = parts[1]
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        payload = json.loads(decoded)
        account_id = payload[_JWT_CLAIM]["chatgpt_account_id"]
        if not isinstance(account_id, str) or not account_id:
            raise ValueError
        return account_id
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeError) as error:
        raise CodexOAuthError("OpenAI Codex token does not contain an account ID") from error
