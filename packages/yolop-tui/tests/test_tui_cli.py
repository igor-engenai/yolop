from importlib.metadata import requires
from pathlib import Path

import pytest
from pydantic_ai import Agent, AgentSpec
from pydantic_ai.models.test import TestModel
from yolop_tui import __file__ as package_file
from yolop_tui.cli import main

from yolop import Yolop


def test_bundled_default_is_a_high_thinking_workspace_coding_agent() -> None:
    spec = AgentSpec.from_file(Path(package_file).parent / "agent_specs" / "coding.yaml")

    assert spec.name == "coding"
    assert spec.model == "openai-codex:gpt-5.6-luna"
    assert spec.model_settings == {"thinking": "high"}
    assert isinstance(spec.instructions, str)
    assert "coding agent" in spec.instructions
    assert [capability.name for capability in spec.capabilities] == [
        "Planning",
        "StuckLoop",
        "WarnNearLimits",
        "Workspace",
    ]
    workspace = next(
        capability for capability in spec.capabilities if capability.name == "Workspace"
    )
    assert workspace.arguments == {
        "shell": True,
        "allowed_commands": [
            "git",
            "rg",
            "grep",
            "find",
            "ls",
            "cat",
            "sed",
            "head",
            "tail",
            "python",
            "uv",
            "pytest",
            "ruff",
            "ty",
            "just",
        ],
    }


def test_bundled_default_constructs_with_installed_capabilities() -> None:
    spec = AgentSpec.from_file(Path(package_file).parent / "agent_specs" / "coding.yaml")
    yolop = Yolop()
    capabilities = yolop.resolve_capabilities(spec)

    Agent.from_spec(
        spec,
        model=TestModel(),
        custom_capability_types=capabilities.selected_types,
        capabilities=capabilities.enforced_capabilities,
    )


def test_tui_distribution_installs_textual_and_the_default_workspace() -> None:
    dependencies = requires("yolop-tui") or []

    assert any(dependency.startswith("textual") for dependency in dependencies)
    assert any(dependency.startswith("yolop-providers") for dependency in dependencies)
    assert any(dependency.startswith("yolop-workspace") for dependency in dependencies)
    assert not any(dependency.startswith("prompt-toolkit") for dependency in dependencies)


def test_cli_injects_the_current_directory_as_workspace(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def capture_run(_spec, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("yolop_tui.cli.run_tui", capture_run)

    main([])

    deps = captured["deps"]
    assert getattr(deps, "workspace") == tmp_path
    assert captured["deps_type"] is type(deps)


def test_cli_starts_bundled_workspace_agent_with_project_sqlite_default(
    tmp_path, monkeypatch
) -> None:
    async def exit_immediately(_spec, **_kwargs) -> None:
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("yolop_tui.cli.run_tui", exit_immediately)

    main([])

    assert (tmp_path / ".yolop" / "runtime.db").is_file()
    assert (tmp_path / ".yolop" / "yolop.log").is_file()


def test_external_agentspec_fully_replaces_the_bundled_default(tmp_path) -> None:
    spec_path = tmp_path / "agent.yaml"
    spec_path.write_text("name: custom\ninstructions: Custom only.\n")

    with pytest.raises(SystemExit, match="must contain a string model"):
        main(["--agent-spec", str(spec_path)])
