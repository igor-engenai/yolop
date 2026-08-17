import stat
from pathlib import Path

from pydantic import SecretStr
from pytest import MonkeyPatch, raises
from yolop_providers import (
    CredentialStatus,
    CredentialStore,
    CredentialStoreError,
    OAuthCredential,
    default_auth_path,
)


def test_oauth_credentials_are_saved_privately_and_report_safe_status(tmp_path) -> None:
    auth_path = tmp_path / "config" / "yolop" / "auth.json"
    store = CredentialStore(auth_path)
    credential = OAuthCredential(
        access_token=SecretStr("access-secret"),
        refresh_token=SecretStr("refresh-secret"),
        expires_at=2_000_000_000.0,
        account_id="account-123",
    )

    store.save_oauth("openai-codex", credential)

    assert store.load_oauth("openai-codex") == credential
    assert store.status("openai-codex") == CredentialStatus(
        authenticated=True,
        expires_at=2_000_000_000.0,
    )
    assert stat.S_IMODE(auth_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600
    assert "access-secret" not in repr(credential)
    assert "refresh-secret" not in repr(credential)


def test_default_auth_path_uses_xdg_config_home(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    assert default_auth_path() == tmp_path / "xdg" / "yolop" / "auth.json"


def test_default_auth_path_rejects_relative_xdg_home(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/config")

    with raises(CredentialStoreError, match="XDG_CONFIG_HOME must be an absolute path"):
        default_auth_path()


def test_logout_removes_only_the_selected_provider_and_is_idempotent(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    first = OAuthCredential(
        access_token=SecretStr("first-access"),
        refresh_token=SecretStr("first-refresh"),
        expires_at=1.0,
        account_id="first-account",
    )
    second = OAuthCredential(
        access_token=SecretStr("second-access"),
        refresh_token=SecretStr("second-refresh"),
        expires_at=2.0,
        account_id="second-account",
    )
    store.save_oauth("openai-codex", first)
    store.save_oauth("future-provider", second)

    assert store.logout("openai-codex") is True
    assert store.logout("openai-codex") is False
    assert store.status("openai-codex") == CredentialStatus(False, None)
    assert store.load_oauth("future-provider") == second


def test_invalid_credential_file_fails_without_revealing_content(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"access_token":"must-not-leak"')
    auth_path.chmod(0o600)

    with raises(CredentialStoreError) as caught:
        CredentialStore(auth_path).status("openai-codex")

    assert str(caught.value) == "Invalid YoloP credential file"
    assert "must-not-leak" not in str(caught.value)


def test_insecure_credential_permissions_are_rejected(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"version": 1, "credentials": {}}')
    auth_path.chmod(0o644)

    with raises(CredentialStoreError, match="permissions must be 0600"):
        CredentialStore(auth_path).status("openai-codex")
