from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.openai import (
    OpenAIModelName,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from .codex import CodexNotAuthenticatedError, CodexOAuth

_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_PROVIDER_NAME = "openai-codex"


class _RefreshingAsyncOpenAI(AsyncOpenAI):
    """Retry one Codex request after refreshing a token rejected with HTTP 401."""

    def __init__(self, *, oauth: CodexOAuth, **kwargs: Any) -> None:
        self._codex_oauth = oauth
        super().__init__(**kwargs)

    async def request(
        self,
        cast_to: Any,
        options: Any,
        *,
        stream: bool = False,
        stream_cls: Any = None,
    ) -> Any:
        stale_access_token = await self._codex_oauth.access_token()
        try:
            return await super().request(
                cast_to,
                options,
                stream=stream,
                stream_cls=stream_cls,
            )
        except Exception as error:
            if getattr(error, "status_code", None) != 401:
                raise
            await self._codex_oauth.refresh_access_token(
                stale_access_token=stale_access_token,
            )
            return await super().request(
                cast_to,
                options,
                stream=stream,
                stream_cls=stream_cls,
            )


class OpenAICodexProvider(OpenAIProvider):
    """Pydantic AI OpenAI Responses provider authenticated by a ChatGPT subscription."""

    @property
    def name(self) -> str:
        return _PROVIDER_NAME

    def __init__(
        self,
        *,
        oauth: CodexOAuth,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        credential = oauth.store.load_oauth(_PROVIDER_NAME)
        if credential is None:
            raise CodexNotAuthenticatedError(
                "OpenAI Codex is not logged in; use /login or `yolop-providers login openai-codex`"
            )
        self._headers = {
            "chatgpt-account-id": credential.account_id,
            "originator": "yolop",
            "OpenAI-Beta": "responses=experimental",
            "User-Agent": _user_agent(),
        }
        self._client = _RefreshingAsyncOpenAI(
            oauth=oauth,
            base_url=_CODEX_BASE_URL,
            api_key=cast(Any, oauth.access_token),
            http_client=http_client,
            default_headers=self._headers,
        )


def create_codex_model(
    model_name: str,
    *,
    oauth: CodexOAuth | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> OpenAIResponsesModel:
    """Create the native Pydantic AI model for one Codex subscription model name."""
    if not model_name:
        raise ValueError("OpenAI Codex model name cannot be empty")
    provider = OpenAICodexProvider(
        oauth=oauth or CodexOAuth(),
        http_client=http_client,
    )
    settings = OpenAIResponsesModelSettings(
        openai_store=False,
        openai_text_verbosity="low",
        openai_reasoning_summary="auto",
        parallel_tool_calls=True,
        extra_headers=provider._headers,
    )
    return OpenAIResponsesModel(
        cast(OpenAIModelName, model_name),
        provider=provider,
        profile=OpenAIModelProfile(openai_supports_prompt_cache_breakpoints=False),
        settings=settings,
    )


def _user_agent() -> str:
    try:
        package_version = version("yolop-providers")
    except PackageNotFoundError:  # pragma: no cover - editable installs have metadata
        package_version = "unknown"
    return f"yolop-providers/{package_version}"
