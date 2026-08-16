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
    TextContent,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.run import EnqueueContent
from rich.console import RenderableType
from yolop_session import (
    ExecutionPin,
    RuntimeSessionSnapshot,
    RuntimeStore,
    SessionConflictError,
    SessionNotFoundError,
    ensure_session_pin,
)

from yolop import Yolop

from .files import FileReferenceError, prepare_prompt
from .rendering import Transcript
from .selection import SelectionOption
from .textual_app import TextualTerminal

_LOGGER = logging.getLogger(__name__)
_SESSION_LOCK_TIMEOUT = 30.0
_TERMINAL_FACTORY = TextualTerminal


def _render_rich_transcript(
    transcript: Transcript,
    _terminal: TextualTerminal,
) -> RenderableType:
    return transcript.renderable()


_TRANSCRIPT_RENDERER = _render_rich_transcript
_HELP_TEXT = (
    "Commands: /new  /resume  /help  /quit\n"
    "Scroll: PageUp/PageDown or mouse wheel · End: newest output"
)


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
    transcript = Transcript.from_messages(session.messages)
    submissions: asyncio.Queue[str] = asyncio.Queue()
    active_turn: _ActiveTurn | None = None
    terminal: TextualTerminal

    def render_transcript() -> None:
        terminal.set_transcript(_TRANSCRIPT_RENDERER(transcript, terminal))

    def refresh_status() -> None:
        input_tokens, output_tokens = _session_usage(session.messages)
        state = active_turn.state if active_turn is not None else "idle"
        queued = (
            f" · queued {active_turn.pending_count}"
            if active_turn is not None and active_turn.pending_count
            else ""
        )
        terminal.set_status(
            f"{_compact(working_directory.name, 14)} · {session.id[:8]} · "
            f"{_compact(pin.model_id, 24)} · ↑{input_tokens} ↓{output_tokens} · {state}{queued}"
        )

    def submit(text: str) -> None:
        command = text.strip()
        if active_turn is None:
            submissions.put_nowait(text)
        elif command == "/help":
            transcript.add_notice(_HELP_TEXT)
            render_transcript()
        elif command.startswith("/"):
            transcript.add_error("Cancel the active run before changing sessions")
            render_transcript()
        else:
            active_turn.enqueue(text)

    def cancel() -> bool:
        if active_turn is None:
            return False
        active_turn.cancel()
        return True

    def set_active(turn: _ActiveTurn | None) -> None:
        nonlocal active_turn
        active_turn = turn
        refresh_status()

    def toggle_tools() -> None:
        transcript.toggle_tools()
        render_transcript()

    def toggle_thinking() -> None:
        transcript.toggle_thinking()
        render_transcript()

    terminal = _TERMINAL_FACTORY(
        on_submit=submit,
        on_cancel=cancel,
        on_toggle_tools=toggle_tools,
        on_toggle_thinking=toggle_thinking,
        cwd=working_directory,
    )
    render_transcript()
    refresh_status()

    async def control() -> None:
        nonlocal session
        while True:
            text = await submissions.get()
            command = text.strip()
            if command == "/quit":
                return
            if command == "/new":
                session = await store.create_session(namespace, pin=pin)
                transcript.reset(session.messages)
                transcript.add_notice(f"New session: {session.id}")
                render_transcript()
                refresh_status()
                continue
            if command == "/resume":
                selected = await terminal.choose(
                    await _session_options(store, namespace=namespace, pin=pin)
                )
                if selected is not None:
                    session = await store.load_session(namespace, selected)
                    ensure_session_pin(session, pin)
                    transcript.reset(session.messages)
                    transcript.add_notice(f"Resumed session: {session.id}")
                    render_transcript()
                    refresh_status()
                continue
            if command == "/help":
                transcript.add_notice(_HELP_TEXT)
                render_transcript()
                continue
            if command.startswith("/"):
                transcript.add_error(f"Unknown command: {command}")
                render_transcript()
                continue
            try:
                prompt = prepare_prompt(text, cwd=working_directory)
            except FileReferenceError as error:
                transcript.add_error(str(error))
                render_transcript()
                continue
            transcript.add_user(text)
            render_transcript()
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
                    render_transcript=render_transcript,
                    refresh_status=refresh_status,
                )
                refresh_status()
            except asyncio.CancelledError:
                raise
            except SessionConflictError:
                session = await store.load_session(namespace, session.id)
                transcript.reset(session.messages)
                transcript.add_error("Session changed; the local run was not saved")
                render_transcript()
                refresh_status()
            except Exception:
                _LOGGER.exception("YoloP terminal run failed for session %s", session.id)
                transcript.reset(session.messages)
                transcript.add_error("Agent run failed")
                render_transcript()
                refresh_status()

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
    terminal: TextualTerminal,
    transcript: Transcript,
    cwd: Path,
    set_active: Callable[["_ActiveTurn | None"], None],
    render_transcript: Callable[[], None],
    refresh_status: Callable[[], None],
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
                            render_transcript=render_transcript,
                            on_change=refresh_status,
                        )
                        set_active(active)
                    else:
                        active.set_context(context)
                    async for event in events:
                        if isinstance(event, EnqueuedMessagesEvent):
                            active.mark_delivered(event.enqueue_id)
                        elif transcript.apply(event):
                            render_transcript()

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
    render_transcript()
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


async def _session_options(
    store: RuntimeStore,
    *,
    namespace: str,
    pin: ExecutionPin,
) -> list[SelectionOption]:
    candidates: list[tuple[float, SelectionOption]] = []
    for session_id in await store.list_sessions(namespace):
        try:
            snapshot = await store.load_session(namespace, session_id)
        except SessionNotFoundError:
            continue
        if snapshot.pin != pin:
            continue
        latest_prompt = ""
        latest_timestamp = 0.0
        for message in reversed(snapshot.messages):
            if not isinstance(message, ModelRequest):
                continue
            user_parts = [part for part in message.parts if isinstance(part, UserPromptPart)]
            if not user_parts:
                continue
            latest_prompt = _display_user_content(user_parts[-1].content)
            latest_timestamp = user_parts[-1].timestamp.timestamp()
            break
        summary = (
            _compact(" ".join(latest_prompt.split()), 48) if latest_prompt else "(empty session)"
        )
        candidates.append(
            (
                latest_timestamp,
                SelectionOption(snapshot.id, f"{snapshot.id}  {summary}"),
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1].value))
    return [option for _timestamp, option in candidates]


def _display_user_content(content: str | Sequence[UserContent]) -> str:
    if isinstance(content, str):
        return content
    for item in content:
        if isinstance(item, str):
            return item
        if isinstance(item, TextContent) and not item.content.startswith("<yolop-file "):
            return item.content
    return ""


def _compact(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"…{value[-(limit - 1) :]}"


def _session_usage(messages: list[ModelMessage]) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for message in messages:
        if isinstance(message, ModelResponse):
            input_tokens += message.usage.input_tokens or 0
            output_tokens += message.usage.output_tokens or 0
    return input_tokens, output_tokens


class _ActiveTurn:
    def __init__(
        self,
        context: RunContext[Any],
        *,
        cwd: Path,
        terminal: TextualTerminal,
        transcript: Transcript,
        render_transcript: Callable[[], None],
        on_change: Callable[[], None],
    ) -> None:
        self._context = context
        self._cwd = cwd
        self._terminal = terminal
        self._transcript = transcript
        self._render_transcript = render_transcript
        self._on_change = on_change
        self._pending: dict[str, str] = {}
        self._cancelling = False

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def state(self) -> str:
        return "cancelling" if self._cancelling else "running"

    def enqueue(self, text: str) -> None:
        try:
            prompt = prepare_prompt(text, cwd=self._cwd)
        except FileReferenceError as error:
            self._transcript.add_error(str(error))
            self._render_transcript()
            return
        content: tuple[EnqueueContent, ...] = (
            (prompt,) if isinstance(prompt, str) else tuple(prompt)
        )
        enqueue_id = self._context.enqueue(*content, priority="asap")
        if enqueue_id is None:
            return
        self._pending[enqueue_id] = text
        self._transcript.add_user(text)
        self._render_transcript()
        self._on_change()

    def set_context(self, context: RunContext[Any]) -> None:
        self._context = context

    def mark_delivered(self, enqueue_id: str) -> None:
        self._pending.pop(enqueue_id, None)
        self._on_change()

    def cancel(self) -> None:
        self._cancelling = True
        self._on_change()
        self._context.cancel()

    def restore_undelivered(self) -> None:
        self._terminal.restore_editor_text("\n\n".join(self._pending.values()))
        self._pending.clear()
        self._on_change()
