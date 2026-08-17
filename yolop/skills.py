from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.resources import files
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import BinaryContent, ModelRetry, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

_BUILTIN_SKILLS = frozenset({"tdd"})


class SkillResource(BaseModel):
    """An immutable manifest entry for one explicit skill resource."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    description: str = ""
    media_type: str = Field(pattern=r"^[^\s/]+/[^\s]+$")
    size: int = Field(ge=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class Skill(BaseModel):
    """An immutable skill snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str
    instructions: str
    resources: tuple[SkillResource, ...] = ()
    source_identity: str = "agent-spec"
    digest: str = ""


ResourceLoader = Callable[[str, str], str | bytes]


@dataclass
class Skills(AbstractCapability[Any]):
    """Expose immutable bundled, inline, and host-library skills to an agent."""

    skills: tuple[Skill, ...] = ()
    resource_loaders: Mapping[str, ResourceLoader] = field(default_factory=dict, repr=False)
    _by_name: dict[str, Skill] = field(init=False, repr=False)
    _toolset: FunctionToolset[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = tuple(_normalize_skill(skill) for skill in self.skills)
        self.skills = normalized
        self._by_name = {skill.name: skill for skill in normalized}
        if len(self._by_name) != len(normalized):
            raise ValueError("Skill names must be unique")
        tools: list[Any] = [Tool(self.load_skill, takes_ctx=False)]
        if any(skill.resources for skill in normalized):
            tools.append(Tool(self.load_skill_resource, takes_ctx=False))
        self._toolset = FunctionToolset(tools, id="skills")

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

    def load_skill_resource(self, skill_name: str, resource_name: str) -> str | BinaryContent:
        """Load one manifest-listed skill resource after explicit model selection."""
        skill = self._by_name.get(skill_name)
        if skill is None:
            raise ModelRetry(f"Unknown skill {skill_name!r}")
        resource = next((item for item in skill.resources if item.name == resource_name), None)
        if resource is None:
            raise ModelRetry(f"Unknown resource {resource_name!r} for skill {skill_name!r}")
        loader = self.resource_loaders.get(f"{skill_name}/{resource_name}")
        if loader is None:
            raise ModelRetry(f"Resource {resource_name!r} is unavailable")
        value = loader(skill_name, resource_name)
        data = value.encode() if isinstance(value, str) else value
        if len(data) != resource.size or sha256(data).hexdigest() != resource.digest:
            raise ModelRetry(f"Resource {resource_name!r} changed; reload the skill snapshot")
        if resource.media_type.startswith("text/"):
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ModelRetry("Text skill resource is not valid UTF-8") from error
        return BinaryContent(data=data, media_type=resource.media_type)


def _normalize_skill(skill: Skill) -> Skill:
    if skill.digest:
        return skill
    canonical = yaml.safe_dump(
        skill.model_dump(mode="json", exclude={"digest"}), sort_keys=True
    ).encode()
    return skill.model_copy(update={"digest": sha256(canonical).hexdigest()})


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
