from collections.abc import AsyncIterator
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic_ai import AgentRunResultEvent, ModelRetry
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pytest import raises
from yolop_runtime import Runtime, SessionPinMismatchError, agent_spec_digest
from yolop_sqlite_session import SQLiteRuntimeStore

from yolop import Yolop
from yolop.skill_libraries import (
    FileSkillLibrary,
    InMemorySkillLibrary,
    SkillDigestConflictError,
    SkillLibraryError,
    SkillResourceError,
    resolve_skill_libraries,
)


async def test_host_skill_library_loads_on_demand_as_an_immutable_snapshot() -> None:
    library = InMemorySkillLibrary(
        identity="company",
        skills=(
            {
                "name": "company-python",
                "description": "Company Python conventions.",
                "instructions": "Use uv, ruff, and ty.",
            },
        ),
    )
    spec = {
        "metadata": {"skill_libraries": {"company": ["company-python"]}},
    }
    resolution = resolve_skill_libraries(spec, {"company": library})

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            assert "company-python" in (info.instructions or "")
            assert [tool.name for tool in info.function_tools] == ["load_skill"]
            yield {
                0: DeltaToolCall(
                    name="load_skill",
                    json_args='{"name":"company-python"}',
                    tool_call_id="load-company-python",
                )
            }
        else:
            assert "Use uv, ruff, and ty" in str(tool_returns[-1].content)
            yield "Used host skill"

    async with Yolop().run(
        spec,
        "Follow company conventions",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        mandatory_capabilities=[resolution.capability],
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Used host skill"
    assert resolution.digest


async def test_skill_resolution_digest_pins_the_runtime_session(tmp_path: Path) -> None:
    first_library = InMemorySkillLibrary(
        identity="company",
        skills=({"name": "python", "description": "one", "instructions": "one"},),
    )
    first = resolve_skill_libraries(
        {"metadata": {"skill_libraries": {"company": ["python"]}}},
        {"company": first_library},
    )
    pinned = first.pinned_spec({"metadata": {"skill_libraries": {"company": ["python"]}}})
    runtime = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    session = await runtime.create_session("tenant", spec=pinned, model_id="test:model")

    second_library = InMemorySkillLibrary(
        identity="company",
        skills=({"name": "python", "description": "two", "instructions": "two"},),
    )
    second = resolve_skill_libraries(
        {"metadata": {"skill_libraries": {"company": ["python"]}}},
        {"company": second_library},
    )
    changed = second.pinned_spec({"metadata": {"skill_libraries": {"company": ["python"]}}})

    assert session.pin.agent_spec_id == agent_spec_digest(pinned)
    assert session.pin.agent_spec_id != agent_spec_digest(changed)
    with raises(SessionPinMismatchError):
        await runtime.run(
            "tenant",
            session.id,
            "use skill",
            spec=changed,
            model=TestModel(),
            model_id="test:model",
            deps=None,
            deps_type=type(None),
            idempotency_key="changed-skill",
        )


def test_file_skill_resources_are_bounded_and_snapshot_checked(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("Use the skill.", encoding="utf-8")
    (tmp_path / "guide.txt").write_text("Guide v1", encoding="utf-8")
    library = FileSkillLibrary(
        identity="company",
        root=tmp_path,
        manifest={
            "python": {
                "description": "Python conventions",
                "instructions": "SKILL.md",
                "resources": {"guide": {"path": "guide.txt", "media_type": "text/plain"}},
            }
        },
    )
    resolution = resolve_skill_libraries(
        {"metadata": {"skill_libraries": {"company": ["python"]}}},
        {"company": library},
    )
    assert resolution.capability.load_skill_resource("python", "guide") == "Guide v1"

    (tmp_path / "guide.txt").write_text("Guide v2", encoding="utf-8")
    with raises(ModelRetry):
        resolution.capability.load_skill_resource("python", "guide")

    (tmp_path / "link.txt").symlink_to(tmp_path / "guide.txt")
    unsafe = FileSkillLibrary(
        identity="unsafe",
        root=tmp_path,
        manifest={
            "unsafe": {
                "description": "unsafe",
                "instructions": "link.txt",
            }
        },
    )
    with raises(SkillResourceError):
        unsafe.list_skills()


def test_file_skill_metadata_cannot_enable_scripts(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("safe", encoding="utf-8")
    library = FileSkillLibrary(
        identity="company",
        root=tmp_path,
        manifest={
            "python": {
                "description": "safe",
                "instructions": "SKILL.md",
                "scripts": ["run.py"],
            }
        },
    )

    with raises(SkillLibraryError):
        library.list_skills()


def test_host_skill_duplicate_names_require_the_same_digest() -> None:
    first = InMemorySkillLibrary(
        identity="first",
        skills=({"name": "shared", "description": "one", "instructions": "one"},),
    )
    second = InMemorySkillLibrary(
        identity="second",
        skills=({"name": "shared", "description": "two", "instructions": "two"},),
    )

    with raises(SkillDigestConflictError):
        resolve_skill_libraries(
            {"metadata": {"skill_libraries": {"first": ["shared"], "second": ["shared"]}}},
            {"first": first, "second": second},
        )


async def test_host_skill_resource_loads_only_from_its_immutable_manifest() -> None:
    resource = b"Use uv."
    library = InMemorySkillLibrary(
        identity="company",
        skills=(
            {
                "name": "company-python",
                "description": "Company Python conventions.",
                "instructions": "Use the company Python skill.",
                "resources": [
                    {
                        "name": "guide",
                        "description": "The guide.",
                        "media_type": "text/plain",
                        "size": len(resource),
                        "digest": sha256(resource).hexdigest(),
                    }
                ],
            },
        ),
        resources={("company-python", "guide"): resource},
    )
    resolution = resolve_skill_libraries(
        {"metadata": {"skill_libraries": {"company": ["company-python"]}}},
        {"company": library},
    )

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            assert [tool.name for tool in info.function_tools] == [
                "load_skill",
                "load_skill_resource",
            ]
            yield {
                0: DeltaToolCall(
                    name="load_skill_resource",
                    json_args='{"skill_name":"company-python","resource_name":"guide"}',
                    tool_call_id="load-guide",
                )
            }
        else:
            assert tool_returns[-1].content == "Use uv."
            yield "Loaded resource"

    async with Yolop().run(
        {"metadata": {"skill_libraries": {"company": ["company-python"]}}},
        "load the guide",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        mandatory_capabilities=[resolution.capability],
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "Loaded resource"


async def test_bundled_skills_are_inactive_until_agent_spec_enables_them() -> None:
    async def respond(
        _messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        assert "tdd" not in (info.instructions or "")
        assert info.function_tools == []
        yield "No skills enabled"

    async with Yolop().run(
        {"instructions": "Base instructions only"},
        "Run without skills",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "No skills enabled"


async def test_agent_spec_enables_and_loads_a_bundled_skill() -> None:
    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            assert "tdd" in (info.instructions or "")
            assert [tool.name for tool in info.function_tools] == ["load_skill"]
            yield {
                0: DeltaToolCall(
                    name="load_skill",
                    json_args='{"name":"tdd"}',
                    tool_call_id="load-tdd",
                )
            }
        else:
            assert "Write one failing test" in str(tool_returns[-1].content)
            yield "Used the bundled skill"

    spec: dict[str, Any] = {
        "capabilities": [
            {"Skills": {"builtin": ["tdd"]}},
        ]
    }

    async with Yolop().run(
        spec,
        "Use test-driven development",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Used the bundled skill"


async def test_agent_spec_carries_an_inline_core_skill_snapshot() -> None:
    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not tool_returns:
            assert "company-python" in (info.instructions or "")
            assert "tdd" not in (info.instructions or "")
            yield {
                0: DeltaToolCall(
                    name="load_skill",
                    json_args='{"name":"company-python"}',
                    tool_call_id="load-company-python",
                )
            }
        else:
            assert "Use uv, ruff, and ty" in str(tool_returns[-1].content)
            yield "Used the Core skill"

    spec: dict[str, Any] = {
        "capabilities": [
            {
                "Skills": {
                    "custom": [
                        {
                            "name": "company-python",
                            "description": "Company Python conventions.",
                            "instructions": "Use uv, ruff, and ty.",
                        }
                    ]
                }
            }
        ]
    }

    async with Yolop().run(
        spec,
        "Follow company conventions",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
    ) as run:
        events = [event async for event in run]

    final_event = events[-1]
    assert isinstance(final_event, AgentRunResultEvent)
    assert final_event.result.output == "Used the Core skill"
