from pathlib import Path

from pydantic_ai import AgentSpec


def test_codex_example_selects_subscription_provider_and_workspace() -> None:
    spec = AgentSpec.from_file(Path("examples/agents/codex.yaml"))

    assert spec.name == "codex-coding"
    assert spec.model == "openai-codex:gpt-5.6-luna"
    assert spec.model_settings == {"thinking": "high"}
    assert [capability.name for capability in spec.capabilities] == ["Workspace"]
