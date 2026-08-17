import asyncio
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic_ai import AgentSpec, AgentStreamEvent, EnqueuedMessagesEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
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
from yolop_runtime import (
    CompactionUnsupportedError,
    ExecutionPin,
    Runtime,
    RuntimeContextSink,
    RuntimeEventSink,
    RuntimeSessionSnapshot,
    RuntimeStore,
    RunTreeNode,
    SessionConflictError,
    SessionNotFoundError,
    ensure_session_pin,
)

from yolop import ProviderCatalog

from .auth import AuthProvider, load_auth_providers
from .files import FileReferenceError, prepare_prompt
from .rendering import Transcript
from .selection import HistoryOption, SelectionOption
from .textual_app import TextualTerminal

_LOGGER = logging.getLogger(__name__)
_TERMINAL_FACTORY = TextualTerminal
_AUTH_PROVIDER_LOADER = load_auth_providers
_PROVIDER_INSTALL_HINT = (
    'No authentication providers are installed. Install with `uv add "yolop[tui,providers]"`.'
)
_HELP_TEXT = (
    "Commands: /new  /resume  /history  /compact [focus]  /goal <condition>  "
    "/goal-status  /goal-stop  /goal-resume  /login  /logout  /help  /quit\n"
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
    provider_catalog: ProviderCatalog | None = None,
    mandatory_capabilities: Sequence[AbstractCapability[Any]] = (),
) -> None:
    """Run the full-screen Textual host until the user exits."""
    runtime = Runtime(store=store, provider_catalog=provider_catalog)
    runtime.kernel.provider_catalog.validate_spec(spec)
    working_directory = (cwd or Path.cwd()).expanduser().resolve()
    pin = _execution_pin(spec, model=model, model_id=model_id)
    session = (
        await runtime.load_session(namespace, session_id)
        if session_id is not None
        else await runtime.create_session(namespace, spec=spec, model_id=pin.model_id)
    )
    ensure_session_pin(session, pin)
    transcript = Transcript.from_messages(session.messages)
    auth_providers: tuple[AuthProvider, ...] | None = None
    submissions: asyncio.Queue[str] = asyncio.Queue()
    active_turn: _ActiveTurn | None = None
    active_goal_id: str | None = None
    auth_state: str | None = None
    terminal: TextualTerminal

    def render_transcript() -> None:
        terminal.set_transcript(transcript.renderable())

    def refresh_status() -> None:
        input_tokens, output_tokens = _session_usage(session.messages)
        state = active_turn.state if active_turn is not None else (auth_state or "idle")
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
        if active_turn is None or active_turn.cancelling:
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
        nonlocal active_goal_id, auth_providers, auth_state, session
        while True:
            text = await submissions.get()
            command = text.strip()
            if command == "/quit":
                return
            if command == "/new":
                session = await runtime.create_session(
                    namespace,
                    spec=spec,
                    model_id=pin.model_id,
                )
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
                    session = await runtime.load_session(namespace, selected)
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
            if command == "/history":
                try:
                    tree = await runtime.list_run_tree(namespace, session_id=session.id)
                    options = _run_history_options(tree)
                    if not options:
                        transcript.add_notice(
                            "No terminal Runs in this Session; "
                            "use /resume to open another saved Session"
                        )
                    else:
                        selected = await terminal.choose_history(options)
                        if selected is not None:
                            action, run_id = selected
                            current = await runtime.load_session(namespace, session.id)
                            if action == "checkout":
                                session = await runtime.checkout(
                                    namespace,
                                    session.id,
                                    run_id,
                                    expected_revision=current.revision,
                                )
                                transcript.reset(session.messages)
                                transcript.add_notice(
                                    f"Checked out Run {run_id[:8]} "
                                    "(session history changed; workspace unchanged)"
                                )
                            elif action == "fork":
                                session = await runtime.fork_session(
                                    namespace,
                                    session.id,
                                    run_id,
                                    expected_revision=current.revision,
                                )
                                transcript.reset(session.messages)
                                active_goal_id = None
                                transcript.add_notice(
                                    f"Forked Session {session.id[:8]} from Run {run_id[:8]} "
                                    "(workspace unchanged)"
                                )
                except SessionConflictError:
                    transcript.add_error("Session changed; history action was not saved")
                except Exception as error:
                    transcript.add_error(f"History action failed: {error}")
                render_transcript()
                refresh_status()
                focus_editor = getattr(terminal, "focus_editor", None)
                if callable(focus_editor):
                    focus_editor()
                continue
            if command == "/goal" or command.startswith("/goal "):
                condition = text.strip()[len("/goal") :].strip()
                if not condition:
                    transcript.add_error("Usage: /goal <condition>")
                    render_transcript()
                    continue
                try:
                    from yolop_deep import GoalRunner

                    goal_model = model or pin.model_id
                    goal_runner = GoalRunner(runtime)
                    evaluator_spec = AgentSpec(
                        model=pin.model_id,
                        name="goal-evaluator",
                        instructions=(
                            "Evaluate the goal only from the supplied transcript evidence. "
                            "Do not call tools. Return met, impossible, or unmet "
                            "with a concrete reason."
                        ),
                        output_schema={
                            "type": "object",
                            "properties": {
                                "verdict": {
                                    "type": "string",
                                    "enum": ["met", "impossible", "unmet"],
                                },
                                "reason": {"type": "string"},
                            },
                            "required": ["verdict", "reason"],
                        },
                    )
                    record = await goal_runner.start(
                        namespace,
                        session.id,
                        goal=condition,
                        spec=spec,
                        model=goal_model,
                        model_id=pin.model_id,
                        evaluator_spec=evaluator_spec,
                        evaluator_model=goal_model,
                        evaluator_model_id=pin.model_id,
                        deps=deps,
                        deps_type=deps_type,
                        max_turns=3,
                    )
                except ImportError:
                    transcript.add_error("Install yolop[deep] to use durable goals")
                except Exception as error:
                    transcript.add_error(f"Goal failed: {error}")
                else:
                    active_goal_id = record.goal_id
                    transcript.add_notice(
                        f"Goal {record.goal_id[:8]}: {record.status.value} "
                        f"({record.reason or 'running'})"
                    )
                render_transcript()
                refresh_status()
                focus_editor = getattr(terminal, "focus_editor", None)
                if callable(focus_editor):
                    focus_editor()
                continue
            if command in {"/goal-status", "/goal-stop", "/goal-resume"}:
                try:
                    from yolop_deep import GoalRunner, GoalStatus

                    goal_runner = GoalRunner(runtime)
                    if active_goal_id is None:
                        records = await goal_runner.list_goals(namespace, session.id)
                        active = [
                            record for record in records if record.status is GoalStatus.ACTIVE
                        ]
                        if active:
                            active_goal_id = active[-1].goal_id
                    if active_goal_id is None:
                        raise ValueError("No goal is selected")
                    if command == "/goal-status":
                        record = await goal_runner.get(namespace, session.id, active_goal_id)
                        if record is None:
                            raise ValueError("Selected goal does not exist")
                    elif command == "/goal-stop":
                        record = await goal_runner.stop(namespace, session.id, active_goal_id)
                    else:
                        record = await goal_runner.resume(
                            namespace,
                            session.id,
                            active_goal_id,
                            spec=spec,
                            model=model or pin.model_id,
                            model_id=pin.model_id,
                            evaluator_model=model or pin.model_id,
                            deps=deps,
                            deps_type=deps_type,
                        )
                    transcript.add_notice(
                        f"Goal {record.goal_id[:8]}: {record.status.value} "
                        f"({record.reason or 'running'})"
                    )
                except ImportError:
                    transcript.add_error("Install yolop[deep] to use durable goals")
                except Exception as error:
                    transcript.add_error(f"Goal command failed: {error}")
                render_transcript()
                refresh_status()
                focus_editor = getattr(terminal, "focus_editor", None)
                if callable(focus_editor):
                    focus_editor()
                continue
            if command == "/compact" or command.startswith("/compact "):
                focus = text.strip()[len("/compact") :].strip() or None
                compactor = _selected_capability(
                    spec,
                    runtime.kernel.provider_catalog,
                    "Compaction",
                )
                try:
                    session = await runtime.compact_session(
                        namespace,
                        session.id,
                        spec=spec,
                        model=model,
                        model_id=pin.model_id,
                        deps=deps,
                        compactor=compactor,
                        focus=focus,
                    )
                except CompactionUnsupportedError as error:
                    transcript.add_error(str(error))
                except SessionConflictError:
                    session = await runtime.load_session(namespace, session.id)
                    transcript.reset(session.messages)
                    transcript.add_error("Session changed; manual compaction was not saved")
                else:
                    transcript.reset(session.messages)
                    transcript.add_notice("Session context compacted")
                render_transcript()
                refresh_status()
                focus_editor = getattr(terminal, "focus_editor", None)
                if callable(focus_editor):
                    focus_editor()
                continue
            if command in {"/login", "/logout"}:
                if auth_providers is None:
                    auth_providers = _AUTH_PROVIDER_LOADER()
                if not auth_providers:
                    transcript.add_error(_PROVIDER_INSTALL_HINT)
                    render_transcript()
                    continue
            if command == "/login":
                assert auth_providers is not None
                selected = await terminal.choose_auth_provider(list(auth_providers))
                if selected is None:
                    continue
                provider = next(
                    provider for provider in auth_providers if provider.name == selected
                )
                auth_state = "logging in"
                refresh_status()
                try:
                    status = await terminal.login_auth_provider(provider)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _LOGGER.warning("Provider login failed for %s", provider.name)
                    transcript.add_error(f"Login failed for {provider.name}")
                else:
                    if status is None:
                        transcript.add_notice(f"Login cancelled for {provider.name}")
                    else:
                        transcript.add_notice(f"Logged in to {provider.name}")
                finally:
                    auth_state = None
                    render_transcript()
                    refresh_status()
                continue
            if command == "/logout":
                assert auth_providers is not None
                selected = await terminal.choose_auth_provider(
                    list(auth_providers),
                    action="Log out",
                )
                if selected is None:
                    continue
                provider = next(
                    provider for provider in auth_providers if provider.name == selected
                )
                try:
                    status = provider.status()
                except Exception:
                    _LOGGER.warning("Provider status failed for %s", provider.name)
                    transcript.add_error(f"Could not read login status for {provider.name}")
                    render_transcript()
                    continue
                if not status.authenticated:
                    transcript.add_notice(f"Already logged out of {provider.name}")
                    render_transcript()
                    continue
                if not await terminal.confirm(f"Log out of {provider.name}?"):
                    continue
                try:
                    removed = provider.logout()
                except Exception:
                    _LOGGER.warning("Provider logout failed for %s", provider.name)
                    transcript.add_error(f"Logout failed for {provider.name}")
                else:
                    message = (
                        f"Logged out of {provider.name}"
                        if removed
                        else f"Already logged out of {provider.name}"
                    )
                    transcript.add_notice(message)
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
                    runtime,
                    spec,
                    prompt=prompt,
                    session=session,
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
                    mandatory_capabilities=mandatory_capabilities,
                )
                refresh_status()
            except asyncio.CancelledError:
                raise
            except SessionConflictError:
                session = await runtime.load_session(namespace, session.id)
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
    runtime: Runtime[DepsT],
    spec: AgentSpec,
    *,
    prompt: str | list[UserContent],
    session: RuntimeSessionSnapshot,
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
    mandatory_capabilities: Sequence[AbstractCapability[Any]],
) -> RuntimeSessionSnapshot:
    active: _ActiveTurn | None = None

    class TurnObserver(RuntimeContextSink, RuntimeEventSink):
        async def set_context(self, context: RunContext[Any]) -> None:
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

        async def emit(self, event: AgentStreamEvent) -> None:
            if active is None:
                return
            if isinstance(event, EnqueuedMessagesEvent):
                active.mark_delivered(event.enqueue_id)
            elif transcript.apply(event):
                render_transcript()

    observer = TurnObserver()
    try:
        completion = await runtime.run(
            namespace,
            session.id,
            prompt,
            spec=spec,
            model=model,
            model_id=pin.model_id,
            deps=deps,
            deps_type=deps_type,
            idempotency_key=f"tui:{uuid4()}",
            event_sink=observer,
            context_sink=observer,
            mandatory_capabilities=mandatory_capabilities,
        )
    finally:
        if active is not None:
            active.restore_undelivered()
        set_active(None)
    transcript.reset(completion.session.messages)
    render_transcript()
    return completion.session


def _run_history_options(tree: Sequence[RunTreeNode]) -> list[HistoryOption]:
    options: list[HistoryOption] = []

    def visit(node: RunTreeNode, depth: int) -> None:
        prompt = " ".join(node.run.prompt.split()) or "(empty prompt)"
        if len(prompt) > 96:
            prompt = f"{prompt[:93]}..."
        tree_prefix = "  " * depth + ("└─ " if depth else "")
        selected_prefix = "▸ " if node.selected else "  "
        label = f" · {node.label}" if node.label else ""
        summary = f"{tree_prefix}{selected_prefix}{prompt} · {node.run.status.value}{label}"
        options.append(HistoryOption(node.run.id, summary, selected=node.selected))
        for child in node.children:
            visit(child, depth + 1)

    for root in tree:
        visit(root, 0)
    return options


def _selected_capability(
    spec: AgentSpec,
    catalog: ProviderCatalog,
    name: str,
) -> Any | None:
    if not catalog.has_capability(name):
        return None
    capability_type = catalog.capability_type(name)
    for capability in spec.capabilities:
        if capability.name == name:
            return capability_type.from_spec(*capability.args, **capability.kwargs)
    return None


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

    @property
    def cancelling(self) -> bool:
        return self._cancelling

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
        self.restore_undelivered()
        self._context.cancel()

    def restore_undelivered(self) -> None:
        if self._pending:
            self._terminal.restore_editor_text("\n\n".join(self._pending.values()))
            self._pending.clear()
            self._on_change()
