from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic_ai import AgentSpec, DeferredToolRequests, ExternalToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pytest import raises
from yolop_runtime import IdempotencyConflictError, Runtime
from yolop_sqlite_session import SQLiteRuntimeStore


@dataclass
class ExternalCapability(AbstractCapability[None]):
    toolset: ExternalToolset

    def get_toolset(self):
        return self.toolset


def external_capability() -> ExternalCapability:
    return ExternalCapability(
        id="external-capability",
        toolset=ExternalToolset(
            [
                ToolDefinition(
                    name="external_lookup",
                    description="Run an external lookup.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                )
            ]
        ),
    )


def model() -> FunctionModel:
    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield "resumed"
        else:
            yield {
                0: DeltaToolCall(
                    name="external_lookup",
                    json_args='{"key":"value"}',
                    tool_call_id="call-1",
                )
            }

    return FunctionModel(stream_function=respond)


def runtime_spec() -> AgentSpec:
    return AgentSpec(model="test:model")


def runtime_kwargs(runtime: Runtime[None], session_id: str) -> dict[str, Any]:
    del runtime, session_id
    return {
        "spec": runtime_spec(),
        "model": model(),
        "model_id": "test:model",
        "deps": None,
        "deps_type": type(None),
        "output_type": [str, DeferredToolRequests],
        "mandatory_capabilities": [external_capability()],
    }


async def test_external_deferred_result_survives_runtime_reload(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    runtime = Runtime(store=SQLiteRuntimeStore(database))
    spec = runtime_spec()
    session = await runtime.create_session("test", spec=spec, model_id="test:model")

    pending = await runtime.run(
        "test",
        session.id,
        "start",
        **runtime_kwargs(runtime, session.id),
        idempotency_key="external-request",
    )
    assert isinstance(pending.run.output, dict)
    assert pending.run.output["calls"][0]["tool_call_id"] == "call-1"

    restarted = Runtime(store=SQLiteRuntimeStore(database))
    resumed = await restarted.resume_deferred_run(
        "test",
        session.id,
        pending.run.id,
        calls={"call-1": "external-result"},
        **runtime_kwargs(restarted, session.id),
        idempotency_key="external-result",
    )

    assert resumed.run.output == "resumed"

    with raises(IdempotencyConflictError):
        await restarted.resume_deferred_run(
            "test",
            session.id,
            pending.run.id,
            calls={"call-1": "different-result"},
            **runtime_kwargs(restarted, session.id),
            idempotency_key="external-result",
        )
