from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import AgentRunResultEvent, AgentSpec
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pytest import raises
from yolop_context import (
    ProjectContextContentError,
    ProjectContextForbiddenFileError,
    ProjectContextLimitError,
    ProjectContextPathError,
    ProjectContextRegistry,
    ProjectContextUnknownFileError,
)

from yolop import Yolop


def test_project_context_loads_only_authorized_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Follow the project rules.", encoding="utf-8")
    registry = ProjectContextRegistry.from_files(
        tmp_path,
        {"agent-rules": "AGENTS.md"},
    )
    spec = AgentSpec(metadata={"project_context": {"files": ["agent-rules"]}})

    composition = registry.build_for_spec(spec)

    assert "Follow the project rules." in composition.instructions
    assert "AGENTS.md" in composition.instructions
    serialized = registry.serialize_config()
    assert serialized["files"] == {"agent-rules": "AGENTS.md"}
    assert "Follow the project rules." not in str(serialized)


async def test_project_context_reaches_the_native_model_as_instructions(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Follow the project rules.", encoding="utf-8")
    registry = ProjectContextRegistry.from_files(tmp_path, {"agent-rules": "AGENTS.md"})
    spec = AgentSpec(metadata={"project_context": {"files": ["agent-rules"]}})
    composition = registry.build_for_spec(spec)

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        assert info.instructions is not None
        assert "Follow the project rules." in info.instructions
        yield "context received"

    async with Yolop().run(
        spec,
        "use context",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        mandatory_capabilities=composition.capabilities,
    ) as run:
        events = [event async for event in run]

    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "context received"


def test_project_context_rejects_traversal_and_symlink_paths(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-project-context.txt"
    outside.write_text("outside", encoding="utf-8")
    with raises(ProjectContextPathError):
        ProjectContextRegistry.from_files(tmp_path, {"escape": "../outside-project-context.txt"})

    target = tmp_path / "real.txt"
    target.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    symlink_registry = ProjectContextRegistry.from_files(tmp_path, {"link": "link.txt"})
    symlink_spec = AgentSpec(metadata={"project_context": {"files": ["link"]}})

    with raises(ProjectContextPathError):
        symlink_registry.build_for_spec(symlink_spec)


def test_project_context_redacts_secret_like_content(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(
        "token=secret-value-1234\nopenai=sk-12345678901234567890\n",
        encoding="utf-8",
    )
    registry = ProjectContextRegistry.from_files(tmp_path, {"notes": "notes.md"})
    spec = AgentSpec(metadata={"project_context": {"files": ["notes"]}})

    composition = registry.build_for_spec(spec)

    assert "secret-value-1234" not in composition.instructions
    assert "sk-12345678901234567890" not in composition.instructions
    assert "[REDACTED]" in composition.instructions


def test_project_context_enforces_text_and_size_limits(tmp_path: Path) -> None:
    large = tmp_path / "large.md"
    large.write_text("12345", encoding="utf-8")
    large_registry = ProjectContextRegistry.from_files(
        tmp_path,
        {"large": "large.md"},
        max_file_bytes=4,
    )
    large_spec = AgentSpec(metadata={"project_context": {"files": ["large"]}})

    with raises(ProjectContextLimitError):
        large_registry.build_for_spec(large_spec)

    (tmp_path / "binary.dat").write_bytes(b"ok\x00binary")
    binary_registry = ProjectContextRegistry.from_files(tmp_path, {"binary": "binary.dat"})
    binary_spec = AgentSpec(metadata={"project_context": {"files": ["binary"]}})
    with raises(ProjectContextContentError):
        binary_registry.build_for_spec(binary_spec)

    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    secret_registry = ProjectContextRegistry.from_files(tmp_path, {"env": ".env"})
    secret_spec = AgentSpec(metadata={"project_context": {"files": ["env"]}})
    with raises(ProjectContextContentError):
        secret_registry.build_for_spec(secret_spec)


def test_project_context_order_and_file_count_are_bounded(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("A", encoding="utf-8")
    (tmp_path / "b.md").write_text("B", encoding="utf-8")
    registry = ProjectContextRegistry.from_files(
        tmp_path,
        {"beta": "b.md", "alpha": "a.md"},
        max_files=2,
    )
    spec = AgentSpec(
        metadata={"project_context": {"files": ["beta", "alpha"]}},
    )
    composition = registry.build_for_spec(spec)

    assert composition.instructions.index('alias="alpha"') < composition.instructions.index(
        'alias="beta"'
    )

    limited = ProjectContextRegistry.from_files(tmp_path, {"a": "a.md", "b": "b.md"}, max_files=1)
    with raises(ProjectContextLimitError):
        limited.build_for_spec(
            spec.model_copy(update={"metadata": {"project_context": {"files": ["a", "b"]}}})
        )


def test_project_context_rejects_forbidden_file_alias(tmp_path: Path) -> None:
    registry = ProjectContextRegistry.from_files(
        tmp_path,
        {"private": "PRIVATE.md"},
        allowed_aliases=set(),
    )
    spec = AgentSpec(metadata={"project_context": {"files": ["private"]}})

    with raises(ProjectContextForbiddenFileError):
        registry.build_for_spec(spec)


def test_project_context_rejects_unknown_file_alias(tmp_path: Path) -> None:
    registry = ProjectContextRegistry.from_files(tmp_path, {})
    spec = AgentSpec(metadata={"project_context": {"files": ["missing"]}})

    with raises(ProjectContextUnknownFileError):
        registry.build_for_spec(spec)
