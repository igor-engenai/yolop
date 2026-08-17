from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError
from pydantic_ai import AgentSpec

from .skills import ResourceLoader, Skill, SkillResource, Skills


class SkillLibraryError(ValueError):
    """A trusted skill library is invalid or unavailable."""

    code = "skill_library_error"


class SkillLibraryNotFoundError(SkillLibraryError):
    """A selected library or skill does not exist."""

    code = "skill_library_not_found"


class SkillDigestConflictError(SkillLibraryError):
    """Two selected libraries provide one name with different immutable content."""

    code = "skill_digest_conflict"


class SkillResourceError(SkillLibraryError):
    """A skill resource is unsafe, missing, changed, or exceeds bounds."""

    code = "skill_resource_error"


class SkillLibrary(Protocol):
    @property
    def identity(self) -> str: ...

    def list_skills(self) -> Sequence[Skill]: ...

    def load_resource(self, skill_name: str, resource_name: str) -> str | bytes: ...


@dataclass(frozen=True)
class SkillLibraryResolution:
    """Immutable selected skills plus the host-built lazy capability."""

    skills: tuple[Skill, ...]
    capability: Skills
    digest: str

    def pinned_spec(self, spec: AgentSpec | Mapping[str, Any]) -> AgentSpec:
        """Return an AgentSpec carrying only the selected immutable digest."""
        validated = spec if isinstance(spec, AgentSpec) else AgentSpec.model_validate(spec)
        metadata = dict(validated.metadata or {})
        metadata["yolop_skill_digest"] = self.digest
        return validated.model_copy(update={"metadata": metadata})


@dataclass(frozen=True)
class InMemorySkillLibrary:
    """A host-owned immutable skill source, useful for local and cloud adapters."""

    identity: str
    skills: tuple[Skill | Mapping[str, Any], ...] = ()
    resources: Mapping[tuple[str, str], str | bytes] = field(default_factory=dict, repr=False)
    _snapshots: tuple[Skill, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = tuple(_source_skill(skill, self.identity) for skill in self.skills)
        names = [skill.name for skill in normalized]
        if len(set(names)) != len(names):
            raise SkillLibraryError("Skill names must be unique in a library")
        object.__setattr__(self, "_snapshots", normalized)
        object.__setattr__(self, "resources", dict(self.resources))

    def list_skills(self) -> tuple[Skill, ...]:
        return self._snapshots

    def load_resource(self, skill_name: str, resource_name: str) -> str | bytes:
        try:
            return self.resources[(skill_name, resource_name)]
        except KeyError as error:
            raise SkillResourceError("Skill resource is unavailable") from error


@dataclass(frozen=True)
class FileSkillLibrary:
    """A bounded trusted file-backed skill library with immutable manifests."""

    identity: str
    root: Path
    manifest: Mapping[str, Mapping[str, Any]]
    max_file_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise SkillLibraryError("Skill library root must be a directory")
        if isinstance(self.max_file_bytes, bool) or not 1 <= self.max_file_bytes <= 1024 * 1024:
            raise SkillLibraryError("Skill library file limit is outside the safe range")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "manifest", dict(self.manifest))

    def list_skills(self) -> tuple[Skill, ...]:
        return tuple(self._snapshot(name, config) for name, config in sorted(self.manifest.items()))

    def load_resource(self, skill_name: str, resource_name: str) -> bytes:
        config = self._skill_config(skill_name)
        resources = config.get("resources", {})
        if not isinstance(resources, Mapping):
            raise SkillResourceError("Skill resources manifest is invalid")
        raw = resources.get(resource_name)
        if not isinstance(raw, Mapping):
            raise SkillResourceError("Skill resource is not manifest-listed")
        path = raw.get("path")
        if not isinstance(path, str):
            raise SkillResourceError("Skill resource path is invalid")
        return self._read(self._safe_path(path))

    def _snapshot(self, name: str, config: Mapping[str, Any]) -> Skill:
        unknown = sorted(set(config) - {"description", "instructions", "resources"})
        if unknown:
            raise SkillLibraryError("Skill manifests cannot define scripts or executable metadata")
        instructions_path = config.get("instructions")
        if not isinstance(instructions_path, str):
            raise SkillLibraryError("Skill instructions must be a relative file path")
        instructions = self._read_text(self._safe_path(instructions_path))
        resources: list[SkillResource] = []
        raw_resources = config.get("resources", {})
        if not isinstance(raw_resources, Mapping):
            raise SkillLibraryError("Skill resources manifest is invalid")
        for resource_name, raw in sorted(raw_resources.items()):
            if not isinstance(resource_name, str) or not isinstance(raw, Mapping):
                raise SkillLibraryError("Skill resource manifest is invalid")
            unknown_resource = sorted(set(raw) - {"path", "description", "media_type"})
            if unknown_resource:
                raise SkillLibraryError("Skill resources cannot define executable metadata")
            path = raw.get("path")
            media_type = raw.get("media_type")
            if not isinstance(path, str) or not isinstance(media_type, str):
                raise SkillLibraryError("Skill resources require path and media_type")
            data = self._read(self._safe_path(path))
            resources.append(
                SkillResource(
                    name=resource_name,
                    description=str(raw.get("description", "")),
                    media_type=media_type,
                    size=len(data),
                    digest=hashlib.sha256(data).hexdigest(),
                )
            )
        return _source_skill(
            {
                "name": name,
                "description": str(config.get("description", "")),
                "instructions": instructions,
                "resources": resources,
            },
            self.identity,
        )

    def _skill_config(self, name: str) -> Mapping[str, Any]:
        config = self.manifest.get(name)
        if config is None:
            raise SkillLibraryNotFoundError(f"Unknown skill {name!r}")
        return config

    def _safe_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise SkillResourceError("Skill paths must be relative and traversal-free")
        candidate = self.root.joinpath(path)
        current = self.root
        for part in path.parts:
            current /= part
            if current.is_symlink():
                raise SkillResourceError("Skill library symlinks are not allowed")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise SkillResourceError("Skill resource is missing") from error
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise SkillResourceError("Skill resource path is outside the library")
        return resolved

    def _read(self, path: Path) -> bytes:
        try:
            with path.open("rb") as source:
                data = source.read(self.max_file_bytes + 1)
        except OSError as error:
            raise SkillResourceError("Skill resource cannot be read") from error
        if len(data) > self.max_file_bytes:
            raise SkillResourceError("Skill resource exceeds the byte limit")
        return data

    def _read_text(self, path: Path) -> str:
        data = self._read(path)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SkillResourceError("Skill instructions must be UTF-8 text") from error


def resolve_skill_libraries(
    spec: AgentSpec | Mapping[str, Any],
    libraries: Mapping[str, SkillLibrary],
) -> SkillLibraryResolution:
    """Resolve AgentSpec library aliases into immutable lazy Skills."""
    metadata = spec.metadata if isinstance(spec, AgentSpec) else spec.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise SkillLibraryError("AgentSpec metadata must be an object")
    raw = metadata.get("skill_libraries") if isinstance(metadata, Mapping) else None
    if raw is None:
        empty = Skills()
        return SkillLibraryResolution((), empty, _digest_skills(()))
    if not isinstance(raw, Mapping):
        raise SkillLibraryError("AgentSpec skill_libraries must be an object")
    selected: dict[str, Skill] = {}
    loaders: dict[str, ResourceLoader] = {}
    for library_alias, raw_names in raw.items():
        library = libraries.get(str(library_alias))
        if library is None:
            raise SkillLibraryNotFoundError(f"Unknown skill library {library_alias!r}")
        if not isinstance(raw_names, (list, tuple)) or not all(
            isinstance(name, str) for name in raw_names
        ):
            raise SkillLibraryError("Skill library selections must contain skill names")
        available = {skill.name: skill for skill in library.list_skills()}
        for name in raw_names:
            skill = available.get(name)
            if skill is None:
                raise SkillLibraryNotFoundError(
                    f"Unknown skill {name!r} in library {library_alias!r}"
                )
            previous = selected.get(name)
            if previous is not None and previous.digest != skill.digest:
                raise SkillDigestConflictError(f"Skill {name!r} has conflicting immutable digests")
            selected[name] = skill
            for resource in skill.resources:
                loaders[f"{name}/{resource.name}"] = library.load_resource
    skills = tuple(selected[name] for name in sorted(selected))
    return SkillLibraryResolution(
        skills, Skills(skills=skills, resource_loaders=loaders), _digest_skills(skills)
    )


def _source_skill(skill: Skill | Mapping[str, Any], identity: str) -> Skill:
    try:
        parsed = skill if isinstance(skill, Skill) else Skill.model_validate(skill)
    except ValidationError as error:
        raise SkillLibraryError("Skill snapshot is invalid") from error
    canonical = json.dumps(
        parsed.model_dump(mode="json", exclude={"digest", "source_identity"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return parsed.model_copy(
        update={"source_identity": identity, "digest": hashlib.sha256(canonical).hexdigest()}
    )


def _digest_skills(skills: Sequence[Skill]) -> str:
    canonical = [
        {
            "name": skill.name,
            "source_identity": skill.source_identity,
            "digest": skill.digest,
        }
        for skill in skills
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
