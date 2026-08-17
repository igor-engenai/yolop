import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from .codex import CodexOAuth, CodexOAuthError, DeviceAuthorization
from .credentials import CredentialStoreError

_PROVIDER = "openai-codex"
_OAUTH_FACTORY: Callable[[], CodexOAuth] = CodexOAuth


def main(argv: Sequence[str] | None = None) -> int:
    """Manage credentials for installed YoloP model providers."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "login":
            return asyncio.run(_login())
        if arguments.command == "status":
            return _status()
        if arguments.command == "logout":
            return _logout()
    except KeyboardInterrupt:
        print("OpenAI Codex login cancelled.", file=sys.stderr)
        return 130
    except (CodexOAuthError, CredentialStoreError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 2  # pragma: no cover - argparse validates commands


def _status() -> int:
    status = _OAUTH_FACTORY().status()
    if not status.authenticated:
        print(f"Not logged in to {_PROVIDER}.")
        return 1
    assert status.expires_at is not None
    expires_at = datetime.fromtimestamp(status.expires_at, tz=UTC).isoformat()
    print(f"Logged in to {_PROVIDER}; access expires at {expires_at}.")
    return 0


def _logout() -> int:
    removed = _OAUTH_FACTORY().logout()
    if removed:
        print(f"Logged out of {_PROVIDER}.")
    else:
        print(f"Already logged out of {_PROVIDER}.")
    return 0


async def _login() -> int:
    oauth = _OAUTH_FACTORY()

    def notify(authorization: DeviceAuthorization) -> None:
        print(f"Open {authorization.verification_uri}")
        print(f"Enter code: {authorization.user_code}")
        print("Waiting for authorization...")

    await oauth.login(notify)
    print(f"Logged in to {_PROVIDER}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage YoloP model provider credentials.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("login", "status", "logout"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("provider", choices=[_PROVIDER])
    return parser
