from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import ModelRetry, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

_BUILTIN_SKILLS = frozenset({"tdd"})


class Skill(BaseModel):
    """An immutable skill snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str
    instructions: str


@dataclass
class Skills(AbstractCapability[Any]):
    """Expose AgentSpec-selected bundled and inline skills to an agent."""

    skills: tuple[Skill, ...] = ()
    _by_name: dict[str, Skill] = field(init=False, repr=False)
    _toolset: FunctionToolset[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {skill.name: skill for skill in self.skills}
        if len(self._by_name) != len(self.skills):
            raise ValueError("Skill names must be unique")
        self._toolset = FunctionToolset(
            [Tool(self.load_skill, takes_ctx=False)],
            id="skills",
        )

    @classmethod
    def from_spec(
        cls,
        *,
        builtin: Sequence[str] = (),
        custom: Sequence[Skill | dict[str, Any]] = (),
    ) -> "Skills":
        skills = [_load_builtin_skill(name) for name in builtin]
        skills.extend(Skill.model_validate(skill) for skill in custom)
        return cls(skills=tuple(skills))

    def get_instructions(self) -> str:
        catalog = "\n".join(f"- {skill.name}: {skill.description}" for skill in self.skills)
        return f"Available skills:\n{catalog}\nCall load_skill with a skill name before using it."

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    def load_skill(self, name: str) -> str:
        """Load the complete instructions for an available skill by name."""
        try:
            skill = self._by_name[name]
        except KeyError as error:
            available = ", ".join(sorted(self._by_name))
            raise ModelRetry(f"Unknown skill {name!r}. Available skills: {available}") from error
        return f"# Skill: {skill.name}\n\n{skill.instructions}"


def _load_builtin_skill(name: str) -> Skill:
    if name not in _BUILTIN_SKILLS:
        available = ", ".join(sorted(_BUILTIN_SKILLS))
        raise ValueError(f"Unknown bundled skill {name!r}. Available skills: {available}")

    content = files("yolop").joinpath("builtin_skills", name, "SKILL.md").read_text()
    parts = content.split("---", maxsplit=2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f"Bundled skill {name!r} has invalid frontmatter")
    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict):
        raise ValueError(f"Bundled skill {name!r} has invalid metadata")
    return Skill.model_validate({**metadata, "instructions": parts[2].strip()})
