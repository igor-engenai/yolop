import asyncio
import base64
import json
from pathlib import Path

import httpx
from pydantic import SecretStr
from pytest import raises
from yolop_providers import (
    CodexNotAuthenticatedError,
    CodexOAuth,
    CodexOAuthError,
    CredentialStore,
    DeviceAuthorization,
    OAuthCredential,
)


def _access_token(account_id: str) -> str:
    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
        }
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


async def _no_sleep(_seconds: float) -> None:
    return None


async def test_device_login_notifies_user_and_saves_oauth_credentials(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    polls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        requests.append(request)
        if request.url.path == "/api/accounts/deviceauth/usercode":
            assert json.loads(request.content) == {"client_id": "app_EMoamEEZ73f0CkXaXp7hrann"}
            return httpx.Response(
                200,
                json={
                    "device_auth_id": "device-id",
                    "user_code": "ABCD-EFGH",
                    "interval": 1,
                },
            )
        if request.url.path == "/api/accounts/deviceauth/token":
            polls += 1
            if polls == 1:
                return httpx.Response(403, json={"error": "authorization_pending"})
            assert json.loads(request.content) == {
                "device_auth_id": "device-id",
                "user_code": "ABCD-EFGH",
            }
            return httpx.Response(
                200,
                json={
                    "authorization_code": "authorization-code",
                    "code_verifier": "code-verifier",
                },
            )
        assert request.url.path == "/oauth/token"
        form = dict(item.split("=", 1) for item in request.content.decode().split("&"))
        assert form == {
            "grant_type": "authorization_code",
            "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
            "code": "authorization-code",
            "code_verifier": "code-verifier",
            "redirect_uri": "https%3A%2F%2Fauth.openai.com%2Fdeviceauth%2Fcallback",
        }
        return httpx.Response(
            200,
            json={
                "access_token": _access_token("account-123"),
                "refresh_token": "refresh-secret",
                "expires_in": 3600,
            },
        )

    store = CredentialStore(tmp_path / "auth.json")
    authorizations: list[DeviceAuthorization] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        base_url="https://auth.openai.com",
    ) as client:
        oauth = CodexOAuth(store=store, http_client=client, sleep=_no_sleep, now=lambda: 1000.0)
        status = await oauth.login(authorizations.append)

    assert status.authenticated is True
    assert status.expires_at == 4600.0
    assert authorizations == [
        DeviceAuthorization(
            verification_uri="https://auth.openai.com/codex/device",
            user_code="ABCD-EFGH",
            expires_in=900.0,
        )
    ]
    credential = store.load_oauth("openai-codex")
    assert credential is not None
    assert credential.access_token.get_secret_value() == _access_token("account-123")
    assert credential.refresh_token.get_secret_value() == "refresh-secret"
    assert credential.account_id == "account-123"
    assert credential.expires_at == 4600.0
    assert polls == 2
    assert len(requests) == 4


async def test_expired_access_token_is_refreshed_and_rotated(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.save_oauth(
        "openai-codex",
        OAuthCredential(
            access_token=SecretStr(_access_token("old-account")),
            refresh_token=SecretStr("old-refresh"),
            expires_at=1000.0,
            account_id="old-account",
        ),
    )
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/oauth/token"
        assert request.content.decode() == (
            "grant_type=refresh_token&refresh_token=old-refresh&"
            "client_id=app_EMoamEEZ73f0CkXaXp7hrann"
        )
        return httpx.Response(
            200,
            json={
                "access_token": _access_token("new-account"),
                "refresh_token": "new-refresh",
                "expires_in": 7200,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        oauth = CodexOAuth(store=store, http_client=client, now=lambda: 1000.0)
        token = await oauth.access_token()

    assert token == _access_token("new-account")
    assert len(requests) == 1
    credential = store.load_oauth("openai-codex")
    assert credential is not None
    assert credential.refresh_token.get_secret_value() == "new-refresh"
    assert credential.expires_at == 8200.0
    assert credential.account_id == "new-account"


async def test_failed_refresh_preserves_the_previous_credential(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    original = OAuthCredential(
        access_token=SecretStr(_access_token("account-123")),
        refresh_token=SecretStr("old-refresh"),
        expires_at=1000.0,
        account_id="account-123",
    )
    store.save_oauth("openai-codex", original)

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="refresh-secret-must-not-leak")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        oauth = CodexOAuth(store=store, http_client=client, now=lambda: 1000.0)
        with raises(CodexOAuthError) as caught:
            await oauth.access_token()

    assert str(caught.value) == "OpenAI Codex token refresh failed (500)"
    assert "refresh-secret" not in str(caught.value)
    assert store.load_oauth("openai-codex") == original


async def test_unexpired_access_token_is_returned_without_a_network_request(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    token = _access_token("account-123")
    store.save_oauth(
        "openai-codex",
        OAuthCredential(
            access_token=SecretStr(token),
            refresh_token=SecretStr("refresh-secret"),
            expires_at=2000.0,
            account_id="account-123",
        ),
    )

    def unexpected(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpired credentials must not make an HTTP request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        oauth = CodexOAuth(store=store, http_client=client, now=lambda: 1000.0)
        assert await oauth.access_token() == token


async def test_access_token_requires_login(tmp_path: Path) -> None:
    oauth = CodexOAuth(store=CredentialStore(tmp_path / "auth.json"))

    with raises(CodexNotAuthenticatedError, match="use /login"):
        await oauth.access_token()


async def test_concurrent_process_clients_rotate_an_expired_refresh_token_once(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    CredentialStore(auth_path).save_oauth(
        "openai-codex",
        OAuthCredential(
            access_token=SecretStr(_access_token("old-account")),
            refresh_token=SecretStr("old-refresh"),
            expires_at=1000.0,
            account_id="old-account",
        ),
    )
    refreshes = 0

    async def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal refreshes
        refreshes += 1
        await asyncio.sleep(0.02)
        return httpx.Response(
            200,
            json={
                "access_token": _access_token("account-123"),
                "refresh_token": "rotated-refresh",
                "expires_in": 3600,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        first = CodexOAuth(
            store=CredentialStore(auth_path),
            http_client=client,
            now=lambda: 1000.0,
        )
        second = CodexOAuth(
            store=CredentialStore(auth_path),
            http_client=client,
            now=lambda: 1000.0,
        )
        tokens = await asyncio.gather(first.access_token(), second.access_token())

    assert tokens == [_access_token("account-123"), _access_token("account-123")]
    assert refreshes == 1


async def test_device_login_honors_slow_down_polling(tmp_path: Path) -> None:
    polls = 0
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.url.path.endswith("/usercode"):
            return httpx.Response(
                200,
                json={"device_auth_id": "device", "user_code": "CODE", "interval": "2"},
            )
        if request.url.path.endswith("/deviceauth/token"):
            polls += 1
            if polls == 1:
                return httpx.Response(400, json={"error": "slow_down"})
            return httpx.Response(
                200,
                json={"authorization_code": "code", "code_verifier": "verifier"},
            )
        return httpx.Response(
            200,
            json={
                "access_token": _access_token("account"),
                "refresh_token": "refresh",
                "expires_in": 3600,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        oauth = CodexOAuth(
            store=CredentialStore(tmp_path / "auth.json"),
            http_client=client,
            sleep=record_sleep,
        )
        await oauth.login(lambda _authorization: None)

    assert sleeps == [2.0, 7.0]


async def test_device_login_times_out_without_saving_credentials(tmp_path: Path) -> None:
    async def wait_forever(_seconds: float) -> None:
        await asyncio.Event().wait()

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/usercode")
        return httpx.Response(
            200,
            json={"device_auth_id": "device", "user_code": "CODE", "interval": 1},
        )

    store = CredentialStore(tmp_path / "auth.json")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        oauth = CodexOAuth(
            store=store,
            http_client=client,
            sleep=wait_forever,
            device_timeout=0.01,
        )
        with raises(CodexOAuthError, match="device login timed out"):
            await oauth.login(lambda _authorization: None)

    assert store.status("openai-codex").authenticated is False


async def test_device_login_propagates_cancellation_without_saving_credentials(
    tmp_path: Path,
) -> None:
    notified = asyncio.Event()

    async def wait_forever(_seconds: float) -> None:
        await asyncio.Event().wait()

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"device_auth_id": "device", "user_code": "CODE", "interval": 1},
        )

    store = CredentialStore(tmp_path / "auth.json")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        oauth = CodexOAuth(store=store, http_client=client, sleep=wait_forever)
        login = asyncio.create_task(oauth.login(lambda _authorization: notified.set()))
        await asyncio.wait_for(notified.wait(), timeout=1)
        login.cancel()
        with raises(asyncio.CancelledError):
            await login

    assert store.status("openai-codex").authenticated is False


async def test_oauth_server_errors_do_not_reveal_response_content(tmp_path: Path) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="provider-secret-must-not-leak")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        oauth = CodexOAuth(store=CredentialStore(tmp_path / "auth.json"), http_client=client)
        with raises(CodexOAuthError) as caught:
            await oauth.login(lambda _authorization: None)

    assert str(caught.value) == "OpenAI Codex device code request failed (500)"
    assert "provider-secret" not in str(caught.value)
