from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness import LLM_API_KEY_ENV_PATTERNS, FileSystem, Shell

_DENIED_ENV_PATTERNS = (*LLM_API_KEY_ENV_PATTERNS, "AZURE_OPENAI_*")

DEFAULT_ALLOWED_COMMANDS: tuple[str, ...] = (
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
)


class WorkspaceDeps(Protocol):
    """Host dependencies required by the Workspace capability."""

    @property
    def workspace(self) -> str | Path: ...


@dataclass
class Workspace(AbstractCapability[Any]):
    """Provide file and optional shell tools for a host-resolved workspace."""

    read_only: bool = False
    shell: bool = False
    allowed_commands: Sequence[str] = DEFAULT_ALLOWED_COMMANDS

    def __post_init__(self) -> None:
        if self.read_only and self.shell:
            raise ValueError("A read-only workspace cannot enable shell commands")

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        root = _workspace_root(ctx.deps)
        capabilities: list[AbstractCapability[Any]] = [
            FileSystem(root_dir=root, read_only=self.read_only)
        ]
        if self.shell:
            capabilities.append(
                Shell(
                    cwd=root,
                    allowed_commands=self.allowed_commands,
                    denied_env_patterns=_DENIED_ENV_PATTERNS,
                )
            )
        return CombinedCapability(capabilities)


def _workspace_root(deps: Any) -> Path:
    value = getattr(deps, "workspace", None)
    if not isinstance(value, (str, Path)):
        raise UserError("Workspace capability requires deps.workspace to be a path")
    root = Path(value).resolve()
    if not root.is_dir():
        raise UserError(f"Workspace directory does not exist: {root}")
    return root
