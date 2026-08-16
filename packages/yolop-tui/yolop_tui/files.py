import json
import os
import re
import shlex
import subprocess
from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from pydantic_ai.messages import TextContent, UserContent

_MAX_FILE_BYTES = 256 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024


class FileReferenceError(ValueError):
    """A user-selected project file cannot be attached safely."""


class FileReferenceCompleter(Completer):
    """Fuzzy-complete project files after an `@` marker."""

    def __init__(self, cwd: Path) -> None:
        self._files = _project_files(cwd.expanduser().resolve())

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        match = re.search(r'(?<!\S)(@(?:"[^"]*|[^\s]*))$', document.text_before_cursor)
        if match is None:
            return
        token = match.group(1)
        query = token[2:] if token.startswith('@"') else token[1:]
        matches = [path for path in self._files if _fuzzy_match(query, path)]
        for path in sorted(matches, key=lambda value: (len(value), value)):
            replacement = (
                f'@"{path}"' if any(character.isspace() for character in path) else f"@{path}"
            )
            yield Completion(replacement, start_position=-len(token), display=path)


def prepare_prompt(text: str, *, cwd: Path) -> str | list[UserContent]:
    """Add bounded project text selected with `@path` to a native user prompt."""
    references = _references(text)
    if not references:
        return text

    root = cwd.expanduser().resolve()
    content: list[UserContent] = [text]
    total_bytes = 0
    attached: set[Path] = set()
    for reference in references:
        relative = Path(reference)
        if relative.is_absolute():
            raise FileReferenceError("File references must be project-relative")
        try:
            path = (root / relative).resolve(strict=True)
        except FileNotFoundError as error:
            raise FileReferenceError(f"File reference does not exist: {reference}") from error
        if not path.is_relative_to(root):
            raise FileReferenceError(f"File reference escapes the project: {reference}")
        if not path.is_file():
            raise FileReferenceError(f"File reference is not a regular file: {reference}")
        if path in attached:
            continue
        attached.add(path)
        with path.open("rb") as file:
            raw = file.read(_MAX_FILE_BYTES + 1)
        if len(raw) > _MAX_FILE_BYTES:
            raise FileReferenceError(f"File reference exceeds 256 KiB: {reference}")
        total_bytes += len(raw)
        if total_bytes > _MAX_TOTAL_BYTES:
            raise FileReferenceError("File references exceed the 1 MiB total limit")
        try:
            file_text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise FileReferenceError(f"File reference is not UTF-8 text: {reference}") from error
        if "\0" in file_text:
            raise FileReferenceError(f"File reference is binary: {reference}")
        display_path = path.relative_to(root).as_posix()
        content.append(
            TextContent(
                content=(
                    f"<yolop-file path={json.dumps(display_path, ensure_ascii=False)}>\n"
                    f"{file_text}\n</yolop-file>"
                )
            )
        )
    return content


def _project_files(root: Path) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=False,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        return tuple(
            sorted(
                path
                for path in result.stdout.decode("utf-8", errors="ignore").split("\0")
                if path and _completion_path_is_safe(root, path)
            )
        )

    files: list[str] = []
    for directory, directories, names in os.walk(root):
        directories[:] = [
            name for name in directories if not name.startswith(".") and name != "__pycache__"
        ]
        base = Path(directory)
        for name in names:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if _completion_path_is_safe(root, relative):
                files.append(relative)
                if len(files) >= 20_000:
                    return tuple(sorted(files))
    return tuple(sorted(files))


def _completion_path_is_safe(root: Path, relative: str) -> bool:
    try:
        path = (root / relative).resolve(strict=True)
    except OSError:
        return False
    return path.is_file() and path.is_relative_to(root)


def _fuzzy_match(query: str, value: str) -> bool:
    characters = iter(value.casefold())
    return all(
        any(candidate == expected for candidate in characters) for expected in query.casefold()
    )


def _references(text: str) -> list[str]:
    try:
        tokens = shlex.split(text)
    except ValueError as error:
        raise FileReferenceError(f"Invalid quoted file reference: {error}") from error
    return [token[1:] for token in tokens if token.startswith("@") and len(token) > 1]
