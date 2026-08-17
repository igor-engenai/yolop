import re
from dataclasses import dataclass
from pathlib import Path

from .files import _project_files

_COMMANDS = ("/new", "/resume", "/login", "/logout", "/help", "/quit")


@dataclass(frozen=True)
class PromptCompletion:
    value: str
    start: int
    display: str


class PromptCompleter:
    """Complete fixed commands and host-authorized project file references."""

    def __init__(self, cwd: Path) -> None:
        self._files = _project_files(cwd.expanduser().resolve())

    def complete(self, text_before_cursor: str) -> tuple[PromptCompletion, ...]:
        current = text_before_cursor.lstrip()
        if current.startswith("/") and not any(character.isspace() for character in current):
            start = len(text_before_cursor) - len(current)
            return tuple(
                PromptCompletion(command, start, command)
                for command in _COMMANDS
                if command.startswith(current) and command != current
            )

        match = re.search(r'(?<!\S)(@(?:"[^"]*|[^\s]*))$', text_before_cursor)
        if match is None:
            return ()
        token = match.group(1)
        query = token[2:] if token.startswith('@"') else token[1:]
        matches = [path for path in self._files if _fuzzy_match(query, path)]
        completions: list[PromptCompletion] = []
        for path in sorted(matches, key=lambda value: (len(value), value)):
            replacement = (
                f'@"{path}"' if any(character.isspace() for character in path) else f"@{path}"
            )
            if replacement != token:
                completions.append(PromptCompletion(replacement, match.start(1), path))
        return tuple(completions)


def _fuzzy_match(query: str, value: str) -> bool:
    characters = iter(value.casefold())
    return all(
        any(candidate == expected for candidate in characters) for expected in query.casefold()
    )
