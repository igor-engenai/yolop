import asyncio
import logging
from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic_ai import AgentSpec, AgentStreamEvent, EnqueuedMessagesEvent, RunContext
from pydantic_ai.exceptions import RunCancelled
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
from pydantic_ai.run import EnqueueContent
from yolop_session import (
    ExecutionPin,
    RuntimeSessionSnapshot,
    RuntimeStore,
    SessionConflictError,
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
    active_turn: _ActiveTurn | None = None

    def submit(text: str) -> None:
        if active_turn is None:
            submissions.put_nowait(text)
        else:
            active_turn.enqueue(text)

    def cancel() -> None:
        if active_turn is not None:
            active_turn.cancel()

    def set_active(turn: _ActiveTurn | None) -> None:
        nonlocal active_turn
        active_turn = turn

    terminal = InlineTerminal(
        on_submit=submit,
        on_cancel=cancel,
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
                    cwd=working_directory,
                    set_active=set_active,
                )
            except asyncio.CancelledError:
                raise
            except SessionConflictError:
                session = await store.load_session(namespace, session.id)
                transcript.reset(session.messages)
                transcript.add_error("Session changed; the local run was not saved")
                terminal.set_transcript(transcript.render())
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
    cwd: Path,
    set_active: Callable[["_ActiveTurn | None"], None],
) -> RuntimeSessionSnapshot:
    async with store.lock_session(
        namespace,
        session.id,
        timeout=_SESSION_LOCK_TIMEOUT,
    ):
        loaded = await store.load_session(namespace, session.id)
        ensure_session_pin(loaded, pin)
        active: _ActiveTurn | None = None
        try:
            try:

                async def handle_events(
                    context: RunContext[DepsT],
                    events: AsyncIterable[AgentStreamEvent],
                ) -> None:
                    nonlocal active
                    if active is None:
                        active = _ActiveTurn(
                            context,
                            cwd=cwd,
                            terminal=terminal,
                            transcript=transcript,
                        )
                        set_active(active)
                    else:
                        active.set_context(context)
                    async for event in events:
                        if isinstance(event, EnqueuedMessagesEvent):
                            active.mark_delivered(event.enqueue_id)
                        elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                            transcript.add_assistant(event.part.content)
                            terminal.set_transcript(transcript.render())
                        elif isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, TextPartDelta
                        ):
                            transcript.add_assistant(event.delta.content_delta)
                            terminal.set_transcript(transcript.render())

                result = await Yolop().execute(
                    spec,
                    prompt,
                    event_stream_handler=handle_events,
                    deps=deps,
                    deps_type=deps_type,
                    model=model,
                    message_history=loaded.messages,
                )
                messages = result.all_messages()
            except RunCancelled as cancelled:
                messages = _cancelled_messages(cancelled)
            saved = await store.replace_session(
                namespace,
                loaded.id,
                expected_revision=loaded.revision,
                messages=messages,
            )
        finally:
            if active is not None:
                active.restore_undelivered()
            set_active(None)
    transcript.reset(saved.messages)
    terminal.set_transcript(transcript.render())
    return saved


def _cancelled_messages(cancelled: RunCancelled) -> list[ModelMessage]:
    messages = cancelled.all_messages()
    last_response_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], ModelResponse)
        ),
        None,
    )
    if last_response_index is None:
        return messages
    response = messages[last_response_index]
    assert isinstance(response, ModelResponse)
    if response.tool_calls and response.state != "interrupted":
        messages[last_response_index] = replace(response, state="interrupted")
    return messages


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


class _ActiveTurn:
    def __init__(
        self,
        context: RunContext[Any],
        *,
        cwd: Path,
        terminal: InlineTerminal,
        transcript: "_Transcript",
    ) -> None:
        self._context = context
        self._cwd = cwd
        self._terminal = terminal
        self._transcript = transcript
        self._pending: dict[str, str] = {}

    def enqueue(self, text: str) -> None:
        try:
            prompt = prepare_prompt(text, cwd=self._cwd)
        except FileReferenceError as error:
            self._transcript.add_error(str(error))
            self._terminal.set_transcript(self._transcript.render())
            return
        content: tuple[EnqueueContent, ...] = (
            (prompt,) if isinstance(prompt, str) else tuple(prompt)
        )
        enqueue_id = self._context.enqueue(*content, priority="asap")
        if enqueue_id is None:
            return
        self._pending[enqueue_id] = text
        self._transcript.add_user(text)
        self._terminal.set_transcript(self._transcript.render())

    def set_context(self, context: RunContext[Any]) -> None:
        self._context = context

    def mark_delivered(self, enqueue_id: str) -> None:
        self._pending.pop(enqueue_id, None)

    def cancel(self) -> None:
        self._context.cancel()

    def restore_undelivered(self) -> None:
        self._terminal.restore_editor_text("\n\n".join(self._pending.values()))
        self._pending.clear()


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
