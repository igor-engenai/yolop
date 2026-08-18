import base64
import json
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr
from pydantic_ai import Agent, AgentRunResultEvent, AgentSpec, Tool
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import CachePoint, ModelResponse, TextPart, ThinkingPart
from pytest import MonkeyPatch, raises
from yolop_providers import (
    CodexNotAuthenticatedError,
    CodexOAuth,
    CredentialStore,
    OAuthCredential,
    create_codex_model,
)

from yolop import Yolop


def _access_token(account_id: str) -> str:
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _text_response(model: str, text: str) -> dict[str, object]:
    return {
        "id": "response-1",
        "created_at": 1,
        "model": model,
        "object": "response",
        "output": [
            {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 14,
        },
    }


async def test_codex_model_uses_oauth_responses_transport_and_native_pydantic_messages(
    tmp_path: Path,
) -> None:
    token = _access_token("account-123")
    store = CredentialStore(tmp_path / "auth.json")
    store.save_oauth(
        "openai-codex",
        OAuthCredential(
            access_token=SecretStr(token),
            refresh_token=SecretStr("refresh-secret"),
            expires_at=2_000_000_000.0,
            account_id="account-123",
        ),
    )
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_text_response("gpt-5.6-luna", "Codex works"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        oauth = CodexOAuth(store=store, now=lambda: 1_000_000_000.0)
        model = create_codex_model("gpt-5.6-luna", oauth=oauth, http_client=client)
        result = await Agent(model, model_settings={"thinking": "high"}).run(
            ["Hello Codex", CachePoint()]
        )

    assert result.output == "Codex works"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 4
    assert len(requests) == 1
    request = requests[0]
    assert request.url == "https://chatgpt.com/backend-api/codex/responses"
    assert request.headers["authorization"] == f"Bearer {token}"
    assert request.headers["chatgpt-account-id"] == "account-123"
    assert request.headers["originator"] == "yolop"
    assert request.headers["openai-beta"] == "responses=experimental"
    assert request.headers["user-agent"].startswith("yolop-providers/")
    body = json.loads(request.content)
    assert body["model"] == "gpt-5.6-luna"
    assert body["store"] is False
    assert body["stream"] is False
    assert body["text"]["verbosity"] == "low"
    assert body["reasoning"]["effort"] == "high"
    assert body["reasoning"]["summary"] == "auto"
    assert body["include"] == ["reasoning.encrypted_content"]
    assert "prompt_cache_breakpoint" not in request.content.decode()


async def test_codex_model_streams_native_thinking_text_and_usage(tmp_path: Path) -> None:
    token = _access_token("account-123")
    store = CredentialStore(tmp_path / "auth.json")
    store.save_oauth(
        "openai-codex",
        OAuthCredential(
            access_token=SecretStr(token),
            refresh_token=SecretStr("refresh-secret"),
            expires_at=2_000_000_000.0,
            account_id="account-123",
        ),
    )
    reasoning = {
        "id": "reasoning-1",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Checked the request"}],
        "encrypted_content": "encrypted-reasoning",
        "status": "completed",
    }
    message = {
        "id": "message-1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": "Streamed Codex",
                "annotations": [],
            }
        ],
    }
    completed = _text_response("gpt-5.6-luna", "Streamed Codex")
    completed["output"] = [reasoning, message]
    stream_events = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                **completed,
                "output": [],
                "status": "in_progress",
                "usage": None,
            },
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {**reasoning, "summary": [], "status": "in_progress"},
        },
        {
            "type": "response.reasoning_summary_part.added",
            "sequence_number": 2,
            "item_id": "reasoning-1",
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 3,
            "item_id": "reasoning-1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "Checked the request",
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "output_index": 0,
            "item": reasoning,
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 5,
            "output_index": 1,
            "item": {**message, "content": [], "status": "in_progress"},
        },
        {
            "type": "response.content_part.added",
            "sequence_number": 6,
            "item_id": "message-1",
            "output_index": 1,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 7,
            "item_id": "message-1",
            "output_index": 1,
            "content_index": 0,
            "delta": "Streamed Codex",
            "logprobs": [],
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 8,
            "output_index": 1,
            "item": message,
        },
        {
            "type": "response.completed",
            "sequence_number": 9,
            "response": completed,
        },
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in stream_events)

    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            text=stream,
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = create_codex_model(
            "gpt-5.6-luna",
            oauth=CodexOAuth(store=store, now=lambda: 1_000_000_000.0),
            http_client=client,
        )
        async with Yolop().run(
            AgentSpec(model="openai-codex:gpt-5.6-luna"),
            "Stream this",
            model=model,
            deps=None,
            deps_type=type(None),
        ) as run:
            events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    response = run.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [part.content for part in response.parts if isinstance(part, ThinkingPart)] == [
        "Checked the request"
    ]
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == [
        "Streamed Codex"
    ]
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4


async def test_codex_model_runs_native_pydantic_tool_loop_and_history(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.save_oauth(
        "openai-codex",
        OAuthCredential(
            access_token=SecretStr(_access_token("account-123")),
            refresh_token=SecretStr("refresh-secret"),
            expires_at=2_000_000_000.0,
            account_id="account-123",
        ),
    )
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            response = _text_response("gpt-5.6-luna", "")
            response["output"] = [
                {
                    "id": "function-1",
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "add",
                    "arguments": '{"left":2,"right":3}',
                    "status": "completed",
                }
            ]
            return httpx.Response(200, json=response)
        return httpx.Response(200, json=_text_response("gpt-5.6-luna", "The result is 5"))

    def add(left: int, right: int) -> int:
        """Add two integers."""
        return left + right

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = create_codex_model(
            "gpt-5.6-luna",
            oauth=CodexOAuth(store=store, now=lambda: 1_000_000_000.0),
            http_client=client,
        )
        result = await Agent(model, tools=[Tool(add)]).run("Add two and three")

    assert result.output == "The result is 5"
    assert len(bodies) == 2
    assert bodies[0]["tools"][0]["name"] == "add"
    second_input = bodies[1]["input"]
    assert isinstance(second_input, list)
    assert any(item.get("type") == "function_call" for item in second_input)
    assert any(
        item.get("type") == "function_call_output" and item.get("output") == "5"
        for item in second_input
    )
    assert len(result.all_messages()) == 4


def test_installed_openai_codex_entry_point_accepts_arbitrary_model_names(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    entry_points = [
        entry_point
        for entry_point in metadata.entry_points(group="yolop.model_providers")
        if entry_point.name == "openai-codex"
    ]

    assert len(entry_points) == 1
    with raises(CodexNotAuthenticatedError, match="use /login"):
        Yolop().run(
            AgentSpec(model="openai-codex:future-codex-model"),
            "Use the provider plugin",
            deps=None,
            deps_type=type(None),
        )


async def test_codex_model_refreshes_after_a_401_and_retries_once(tmp_path: Path) -> None:
    old_token = _access_token("account-123")
    new_token = _access_token("account-456")
    store = CredentialStore(tmp_path / "auth.json")
    store.save_oauth(
        "openai-codex",
        OAuthCredential(
            access_token=SecretStr(old_token),
            refresh_token=SecretStr("old-refresh"),
            expires_at=2_000_000_000.0,
            account_id="account-123",
        ),
    )
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "auth.openai.com":
            return httpx.Response(
                200,
                json={
                    "access_token": new_token,
                    "refresh_token": "new-refresh",
                    "expires_in": 3_600,
                },
            )
        if request.headers["authorization"] == f"Bearer {old_token}":
            return httpx.Response(401, json={"error": {"code": "token_expired"}})
        return httpx.Response(200, json=_text_response("gpt-5.6-luna", "retried"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        oauth = CodexOAuth(
            store=store,
            http_client=client,
            now=lambda: 1_000_000_000.0,
        )
        model = create_codex_model("gpt-5.6-luna", oauth=oauth, http_client=client)
        result = await Agent(model).run("Retry Codex")

    assert result.output == "retried"
    response_requests = [request for request in requests if request.url.host == "chatgpt.com"]
    assert [request.headers["authorization"] for request in response_requests] == [
        f"Bearer {old_token}",
        f"Bearer {new_token}",
    ]
    refreshed = store.load_oauth("openai-codex")
    assert refreshed is not None
    assert refreshed.access_token.get_secret_value() == new_token


async def test_codex_model_does_not_refresh_after_a_non_401_error(tmp_path: Path) -> None:
    token = _access_token("account-123")
    store = CredentialStore(tmp_path / "auth.json")
    store.save_oauth(
        "openai-codex",
        OAuthCredential(
            access_token=SecretStr(token),
            refresh_token=SecretStr("refresh-secret"),
            expires_at=2_000_000_000.0,
            account_id="account-123",
        ),
    )
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "chatgpt.com"
        return httpx.Response(403, json={"error": {"code": "forbidden"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        oauth = CodexOAuth(
            store=store,
            http_client=client,
            now=lambda: 1_000_000_000.0,
        )
        model = create_codex_model("gpt-5.6-luna", oauth=oauth, http_client=client)
        with raises(ModelHTTPError) as error:
            await Agent(model).run("Do not refresh")

    assert error.value.status_code == 403
    assert len(requests) == 1


async def test_codex_model_refreshes_before_sending_a_model_request(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.save_oauth(
        "openai-codex",
        OAuthCredential(
            access_token=SecretStr(_access_token("account-123")),
            refresh_token=SecretStr("old-refresh"),
            expires_at=1000.0,
            account_id="account-123",
        ),
    )
    refreshed_token = _access_token("account-123") + "-refreshed"
    auth_requests = 0

    def refresh(request: httpx.Request) -> httpx.Response:
        nonlocal auth_requests
        auth_requests += 1
        assert request.url == "https://auth.openai.com/oauth/token"
        return httpx.Response(
            200,
            json={
                "access_token": refreshed_token,
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {refreshed_token}"
        return httpx.Response(200, json=_text_response("future-model", "refreshed"))

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(refresh)) as auth_client,
        httpx.AsyncClient(transport=httpx.MockTransport(respond)) as model_client,
    ):
        oauth = CodexOAuth(
            store=store,
            http_client=auth_client,
            now=lambda: 1000.0,
        )
        model = create_codex_model("future-model", oauth=oauth, http_client=model_client)
        result = await Agent(model).run("Use the refreshed token")

    assert result.output == "refreshed"
    assert auth_requests == 1
