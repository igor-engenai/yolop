from collections.abc import Iterable
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from .files import FileReferenceCompleter

_COMMANDS = ("/new", "/resume", "/help", "/quit")


class TuiCompleter(Completer):
    """Complete the fixed command set and project file references."""

    def __init__(self, cwd: Path) -> None:
        self._files = FileReferenceCompleter(cwd)

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        current = document.text_before_cursor.lstrip()
        if current.startswith("/") and not any(character.isspace() for character in current):
            for command in _COMMANDS:
                if command.startswith(current):
                    yield Completion(command, start_position=-len(current))
            return
        yield from self._files.get_completions(document, complete_event)
