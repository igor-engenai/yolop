from pathlib import Path

from pydantic_ai import AgentSpec
from pydantic_ai.models import KnownModelName, Model
from yolop_session import RuntimeStore

from .files import FileReferenceCompleter
from .terminal import InlineTerminal


async def run_tui[DepsT](
    spec: AgentSpec,
    *,
    store: RuntimeStore,
    namespace: str,
    deps: DepsT,
    deps_type: type[DepsT],
    model: Model | KnownModelName | str | None = None,
    model_id: str | None = None,
    session_id: str | None = None,
    cwd: Path | None = None,
) -> None:
    """Run the inline terminal host until the user exits."""
    _ = (spec, store, namespace, deps, deps_type, model, model_id, session_id)
    working_directory = (cwd or Path.cwd()).expanduser().resolve()
    terminal: InlineTerminal

    def submit(text: str) -> None:
        if text.strip() == "/quit":
            terminal.stop()

    terminal = InlineTerminal(
        on_submit=submit,
        completer=FileReferenceCompleter(working_directory),
    )
    await terminal.run()
