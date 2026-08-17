import sys
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai import AgentRunResultEvent, AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pytest import raises
from yolop_mcp import (
    MCPFeatureForbiddenError,
    MCPForbiddenAliasError,
    MCPRegistry,
    MCPServerConfig,
    MCPToolPrefixConflictError,
    MCPUnknownAliasError,
)

from yolop import Yolop


def test_stdio_server_config_is_host_owned_and_rejects_model_credentials() -> None:
    with raises(ValidationError, match="model credential"):
        MCPServerConfig(
            alias="local",
            transport="stdio",
            command="python",
            environment={"OPENAI_API_KEY": "must-not-pass"},
        )
    with raises(ValidationError, match="model credential"):
        MCPServerConfig(
            alias="local",
            transport="stdio",
            command="python",
            environment_secret_refs={"OPENAI_API_KEY": "secret/model-key"},
        )


async def test_local_stdio_alias_exposes_a_native_prefixed_tool(tmp_path: Path) -> None:
    server = tmp_path / "server.py"
    server.write_text(
        """
import os
from mcp.server.fastmcp import FastMCP

server = FastMCP('local-test')

@server.tool()
def echo(value: str) -> str:
    return f"echo:{value}:{os.getenv('MCP_VISIBLE')}:{os.getenv('OPENAI_API_KEY')}"

server.run(transport='stdio')
""",
        encoding="utf-8",
    )
    registry = MCPRegistry(
        [
            MCPServerConfig(
                alias="local",
                transport="stdio",
                command=sys.executable,
                args=(str(server),),
                timeout_seconds=5,
                environment={"MCP_VISIBLE": "visible"},
            )
        ]
    )
    spec = AgentSpec(metadata={"mcp": {"servers": [{"alias": "local"}]}})
    composition = await registry.build_for_spec(spec)

    async def respond(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            yield {
                0: DeltaToolCall(
                    name="local_echo",
                    json_args='{"value":"hello"}',
                    tool_call_id="local-call",
                )
            }
        else:
            assert returns[-1].content == "echo:hello:visible:None"
            yield "native MCP tool completed"

    try:
        async with Yolop().run(
            spec,
            "use the local tool",
            model=FunctionModel(stream_function=respond),
            deps=None,
            deps_type=type(None),
            mandatory_capabilities=composition.capabilities,
        ) as run:
            events = [event async for event in run]
    finally:
        await composition.aclose()

    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "native MCP tool completed"


async def test_unknown_and_forbidden_aliases_fail_before_capability_build() -> None:
    registry = MCPRegistry(
        [MCPServerConfig(alias="local", transport="stdio", command="python")],
        allowed_aliases={"local"},
    )
    unknown = AgentSpec(metadata={"mcp": {"servers": [{"alias": "missing"}]}})
    forbidden_registry = MCPRegistry(
        [MCPServerConfig(alias="local", transport="stdio", command="python")],
        allowed_aliases=set(),
    )
    forbidden = AgentSpec(metadata={"mcp": {"servers": [{"alias": "local"}]}})

    with raises(MCPUnknownAliasError):
        await registry.build_for_spec(unknown)
    with raises(MCPForbiddenAliasError):
        await forbidden_registry.build_for_spec(forbidden)


async def test_secret_references_are_resolved_in_memory_only() -> None:
    config = MCPServerConfig(
        alias="remote",
        transport="http",
        url="https://mcp.example.test/mcp",
        header_secret_refs={"Authorization": "secret/mcp-token"},
        environment_secret_refs={},
    )
    registry = MCPRegistry([config])
    spec = AgentSpec(metadata={"mcp": {"servers": [{"alias": "remote"}]}})
    resolved: list[str] = []

    async def resolve(reference: str) -> str:
        resolved.append(reference)
        return "Bearer secret-value"

    composition = await registry.build_for_spec(spec, secret_resolver=resolve)

    assert resolved == ["secret/mcp-token"]
    assert "secret-value" not in config.model_dump_json()
    await composition.aclose()


async def test_resources_and_skills_require_both_host_and_agent_opt_in() -> None:
    config = MCPServerConfig(
        alias="remote",
        transport="http",
        url="https://mcp.example.test/mcp",
        allow_resources=True,
        allow_skills=True,
    )
    registry = MCPRegistry([config])
    spec = AgentSpec(metadata={"mcp": {"servers": [{"alias": "remote"}]}})
    composition = await registry.build_for_spec(spec)

    with raises(MCPFeatureForbiddenError):
        await composition.connections[0].list_resources()
    with raises(MCPFeatureForbiddenError):
        await composition.connections[0].list_skills()
    await composition.aclose()


async def test_native_initialization_error_closes_the_transport() -> None:
    registry = MCPRegistry(
        [
            MCPServerConfig(
                alias="unavailable",
                transport="http",
                url="http://127.0.0.1:1/mcp",
                timeout_seconds=1,
            )
        ]
    )
    spec = AgentSpec(metadata={"mcp": {"servers": [{"alias": "unavailable"}]}})
    composition = await registry.build_for_spec(spec)
    native = composition.connections[0].toolset

    with raises(Exception):
        await native.list_tools()

    assert not native.is_running
    await composition.aclose()


async def test_host_prefixes_must_be_unique_before_transport_start() -> None:
    registry = MCPRegistry(
        [
            MCPServerConfig(
                alias="one",
                transport="http",
                url="https://one.example.test/mcp",
                tool_prefix="shared",
            ),
            MCPServerConfig(
                alias="two",
                transport="http",
                url="https://two.example.test/mcp",
                tool_prefix="shared",
            ),
        ]
    )
    spec = AgentSpec(
        metadata={
            "mcp": {
                "servers": [{"alias": "one"}, {"alias": "two"}],
            }
        }
    )

    with raises(MCPToolPrefixConflictError):
        await registry.build_for_spec(spec)
