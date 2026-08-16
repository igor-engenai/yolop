import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import AgentSpec
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextContent,
    TextPart,
    TextPartDelta,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model
from yolop_session import (
    ExecutionPin,
    RuntimeSessionSnapshot,
    RuntimeStore,
    ensure_session_pin,
)

from yolop import Yolop

from .files import FileReferenceCompleter, FileReferenceError, prepare_prompt
from .terminal import InlineTerminal

_LOGGER = logging.getLogger(__name__)
_SESSION_LOCK_TIMEOUT = 30.0


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
    working_directory = (cwd or Path.cwd()).expanduser().resolve()
    pin = _execution_pin(spec, model=model, model_id=model_id)
    session = (
        await store.load_session(namespace, session_id)
        if session_id is not None
        else await store.create_session(namespace, pin=pin)
    )
    ensure_session_pin(session, pin)
    transcript = _Transcript.from_messages(session.messages)
    submissions: asyncio.Queue[str] = asyncio.Queue()
    terminal = InlineTerminal(
        on_submit=submissions.put_nowait,
        completer=FileReferenceCompleter(working_directory),
    )
    terminal.set_transcript(transcript.render())

    async def control() -> None:
        nonlocal session
        while True:
            text = await submissions.get()
            if text.strip() == "/quit":
                return
            try:
                prompt = prepare_prompt(text, cwd=working_directory)
            except FileReferenceError as error:
                transcript.add_error(str(error))
                terminal.set_transcript(transcript.render())
                continue
            transcript.add_user(text)
            terminal.set_transcript(transcript.render())
            try:
                session = await _run_turn(
                    spec,
                    prompt=prompt,
                    session=session,
                    store=store,
                    namespace=namespace,
                    pin=pin,
                    deps=deps,
                    deps_type=deps_type,
                    model=model,
                    terminal=terminal,
                    transcript=transcript,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("YoloP terminal run failed for session %s", session.id)
                transcript.reset(session.messages)
                transcript.add_error("Agent run failed")
                terminal.set_transcript(transcript.render())

    terminal_task = asyncio.create_task(terminal.run(), name="yolop-tui-terminal")
    control_task = asyncio.create_task(control(), name="yolop-tui-controller")
    done, _pending = await asyncio.wait(
        {terminal_task, control_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if control_task in done:
        await control_task
        terminal.stop()
        await terminal_task
    else:
        await terminal_task
        control_task.cancel()
        await asyncio.gather(control_task, return_exceptions=True)


async def _run_turn[DepsT](
    spec: AgentSpec,
    *,
    prompt: str | list[UserContent],
    session: RuntimeSessionSnapshot,
    store: RuntimeStore,
    namespace: str,
    pin: ExecutionPin,
    deps: DepsT,
    deps_type: type[DepsT],
    model: Model | KnownModelName | str | None,
    terminal: InlineTerminal,
    transcript: "_Transcript",
) -> RuntimeSessionSnapshot:
    async with store.lock_session(
        namespace,
        session.id,
        timeout=_SESSION_LOCK_TIMEOUT,
    ):
        loaded = await store.load_session(namespace, session.id)
        ensure_session_pin(loaded, pin)
        async with Yolop().run(
            spec,
            prompt,
            deps=deps,
            deps_type=deps_type,
            model=model,
            message_history=loaded.messages,
        ) as run:
            async for event in run:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    transcript.add_assistant(event.part.content)
                    terminal.set_transcript(transcript.render())
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    transcript.add_assistant(event.delta.content_delta)
                    terminal.set_transcript(transcript.render())
        assert run.result is not None
        saved = await store.replace_session(
            namespace,
            loaded.id,
            expected_revision=loaded.revision,
            messages=run.all_messages(),
        )
    transcript.reset(saved.messages)
    terminal.set_transcript(transcript.render())
    return saved


def _execution_pin(
    spec: AgentSpec,
    *,
    model: Model | KnownModelName | str | None,
    model_id: str | None,
) -> ExecutionPin:
    resolved_id = model_id
    if resolved_id is None and isinstance(model, str):
        resolved_id = model
    if resolved_id is None and model is None and isinstance(spec.model, str):
        resolved_id = spec.model
    if not resolved_id:
        raise ValueError("model_id is required when the resolved model is not a string reference")
    return ExecutionPin.from_spec(spec, model_id=resolved_id)


class _Transcript:
    def __init__(self) -> None:
        self._entries: list[tuple[str, str]] = []

    @classmethod
    def from_messages(cls, messages: list[ModelMessage]) -> "_Transcript":
        transcript = cls()
        transcript.reset(messages)
        return transcript

    def reset(self, messages: list[ModelMessage]) -> None:
        self._entries = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        text = _display_user_content(part.content)
                        if text:
                            self._entries.append(("user", text))
            elif isinstance(message, ModelResponse):
                text = "".join(part.content for part in message.parts if isinstance(part, TextPart))
                if text:
                    self._entries.append(("assistant", text))

    def add_user(self, text: str) -> None:
        self._entries.append(("user", text))

    def add_assistant(self, text: str) -> None:
        if self._entries and self._entries[-1][0] == "assistant":
            role, current = self._entries[-1]
            self._entries[-1] = (role, current + text)
        else:
            self._entries.append(("assistant", text))

    def add_error(self, text: str) -> None:
        self._entries.append(("error", text))

    def render(self) -> str:
        blocks = []
        for role, text in self._entries:
            if role == "user":
                blocks.append(f"› {text}")
            elif role == "error":
                blocks.append(f"Error: {text}")
            else:
                blocks.append(text)
        return "\n\n".join(blocks)


def _display_user_content(content: str | Sequence[UserContent]) -> str:
    if isinstance(content, str):
        return content
    for item in content:
        if isinstance(item, str):
            return item
        if isinstance(item, TextContent) and not item.content.startswith("<yolop-file "):
            return item.content
    return ""
