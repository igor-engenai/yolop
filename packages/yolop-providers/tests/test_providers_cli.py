from pytest import CaptureFixture, MonkeyPatch
from yolop_providers import CodexOAuthError, CredentialStatus, DeviceAuthorization, cli


class _LoginOAuth:
    async def login(self, notify) -> CredentialStatus:
        notify(
            DeviceAuthorization(
                verification_uri="https://auth.openai.com/codex/device",
                user_code="ABCD-EFGH",
                expires_in=900,
            )
        )
        return CredentialStatus(authenticated=True, expires_at=2_000_000_000.0)


def test_cli_login_prints_device_instructions_without_tokens(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_OAUTH_FACTORY", _LoginOAuth)

    exit_code = cli.main(["login", "openai-codex"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "https://auth.openai.com/codex/device" in captured.out
    assert "ABCD-EFGH" in captured.out
    assert "Waiting for authorization" in captured.out
    assert "Logged in to openai-codex" in captured.out
    assert "token" not in captured.out.casefold()
    assert captured.err == ""


def test_cli_status_reports_safe_expiry(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    class StatusOAuth:
        def status(self) -> CredentialStatus:
            return CredentialStatus(authenticated=True, expires_at=2_000_000_000.0)

    monkeypatch.setattr(cli, "_OAUTH_FACTORY", StatusOAuth)

    assert cli.main(["status", "openai-codex"]) == 0

    captured = capsys.readouterr()
    assert "Logged in to openai-codex" in captured.out
    assert "2033-05-18T03:33:20+00:00" in captured.out
    assert "secret" not in captured.out.casefold()
    assert captured.err == ""


def test_cli_status_returns_one_when_logged_out(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    class StatusOAuth:
        def status(self) -> CredentialStatus:
            return CredentialStatus(authenticated=False, expires_at=None)

    monkeypatch.setattr(cli, "_OAUTH_FACTORY", StatusOAuth)

    assert cli.main(["status", "openai-codex"]) == 1
    assert capsys.readouterr().out == "Not logged in to openai-codex.\n"


def test_cli_logout_is_idempotent(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    logged_in = True

    class LogoutOAuth:
        def logout(self) -> bool:
            nonlocal logged_in
            removed = logged_in
            logged_in = False
            return removed

    monkeypatch.setattr(cli, "_OAUTH_FACTORY", LogoutOAuth)

    assert cli.main(["logout", "openai-codex"]) == 0
    assert cli.main(["logout", "openai-codex"]) == 0

    assert capsys.readouterr().out == (
        "Logged out of openai-codex.\nAlready logged out of openai-codex.\n"
    )


def test_cli_login_reports_safe_oauth_error(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    class FailingOAuth:
        async def login(self, _notify) -> CredentialStatus:
            raise CodexOAuthError("OpenAI Codex device login failed")

    monkeypatch.setattr(cli, "_OAUTH_FACTORY", FailingOAuth)

    assert cli.main(["login", "openai-codex"]) == 1
    assert capsys.readouterr().err == "Error: OpenAI Codex device login failed\n"


def test_cli_login_ctrl_c_returns_shell_interrupt_status(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    class CancelledOAuth:
        async def login(self, _notify) -> CredentialStatus:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_OAUTH_FACTORY", CancelledOAuth)

    assert cli.main(["login", "openai-codex"]) == 130
    assert capsys.readouterr().err == "OpenAI Codex login cancelled.\n"
