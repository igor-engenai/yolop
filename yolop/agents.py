from importlib.resources import files

from pydantic_ai import AgentSpec


def coding_agent_spec() -> AgentSpec:
    """Load a fresh copy of the built-in coding AgentSpec."""
    content = files("yolop").joinpath("agent_specs", "coding.yaml").read_text()
    return AgentSpec.from_text(content)
