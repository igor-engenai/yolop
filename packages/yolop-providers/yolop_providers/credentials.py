import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator

_DOCUMENT_VERSION = 1


class CredentialStoreError(ValueError):
    """A safe error raised when the credential store cannot be used."""


class OAuthCredential(BaseModel):
    """Stored OAuth tokens whose representation never reveals token values."""

    model_config = ConfigDict(frozen=True)

    access_token: SecretStr
    refresh_token: SecretStr
    expires_at: float
    account_id: str

    @field_validator("access_token", "refresh_token")
    @classmethod
    def _token_is_not_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("OAuth tokens cannot be empty")
        return value

    @field_validator("account_id")
    @classmethod
    def _account_id_is_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("OAuth account ID cannot be empty")
        return value


@dataclass(frozen=True)
class CredentialStatus:
    authenticated: bool
    expires_at: float | None


class CredentialStore:
    """Persist provider credentials in one private, atomically replaced JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or default_auth_path()).expanduser()
        self._lock = FileLock(f"{self.path}.lock")

    def save_oauth(self, provider: str, credential: OAuthCredential) -> None:
        _validate_provider(provider)
        self._ensure_directory()
        with self._lock:
            credentials = self._read_credentials()
            credentials[provider] = credential
            self._write_credentials(credentials)

    def load_oauth(self, provider: str) -> OAuthCredential | None:
        _validate_provider(provider)
        if not self.path.exists():
            return None
        self._ensure_directory()
        with self._lock:
            return self._read_credentials().get(provider)

    def status(self, provider: str) -> CredentialStatus:
        credential = self.load_oauth(provider)
        return CredentialStatus(
            authenticated=credential is not None,
            expires_at=credential.expires_at if credential is not None else None,
        )

    def logout(self, provider: str) -> bool:
        _validate_provider(provider)
        if not self.path.exists():
            return False
        self._ensure_directory()
        with self._lock:
            credentials = self._read_credentials()
            removed = credentials.pop(provider, None) is not None
            if removed:
                self._write_credentials(credentials)
            return removed

    def _ensure_directory(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)

    def _read_credentials(self) -> dict[str, OAuthCredential]:
        if not self.path.exists():
            return {}
        _check_private_file(self.path)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") != _DOCUMENT_VERSION:
                raise CredentialStoreError("Unsupported YoloP credential file version")
            raw_credentials = raw.get("credentials")
            if not isinstance(raw_credentials, dict):
                raise CredentialStoreError("Invalid YoloP credential file")
            return {
                provider: OAuthCredential.model_validate(value)
                for provider, value in raw_credentials.items()
                if isinstance(provider, str)
            }
        except CredentialStoreError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, TypeError) as error:
            raise CredentialStoreError("Invalid YoloP credential file") from error

    def _write_credentials(self, credentials: dict[str, OAuthCredential]) -> None:
        document = {
            "version": _DOCUMENT_VERSION,
            "credentials": {
                provider: _credential_json(credential)
                for provider, credential in sorted(credentials.items())
            },
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
                json.dump(document, temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
            self.path.chmod(0o600)
        finally:
            temporary_path.unlink(missing_ok=True)


def default_auth_path() -> Path:
    configured = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(configured).expanduser() if configured else Path.home() / ".config"
    if not config_home.is_absolute():
        raise CredentialStoreError("XDG_CONFIG_HOME must be an absolute path")
    return config_home / "yolop" / "auth.json"


def _credential_json(credential: OAuthCredential) -> dict[str, Any]:
    return {
        "access_token": credential.access_token.get_secret_value(),
        "refresh_token": credential.refresh_token.get_secret_value(),
        "expires_at": credential.expires_at,
        "account_id": credential.account_id,
    }


def _check_private_file(path: Path) -> None:
    file_stat = path.lstat()
    if not stat.S_ISREG(file_stat.st_mode) or path.is_symlink():
        raise CredentialStoreError("YoloP credential path must be a regular file")
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise CredentialStoreError("YoloP credential file permissions must be 0600")


def _validate_provider(provider: str) -> None:
    if not provider or provider != provider.strip():
        raise ValueError("Provider name cannot be empty or contain outer whitespace")
