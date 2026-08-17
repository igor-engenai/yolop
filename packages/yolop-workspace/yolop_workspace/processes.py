from __future__ import annotations

import asyncio
import fnmatch
import inspect
import os
import re
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic_ai_harness import LLM_API_KEY_ENV_PATTERNS
from yolop_runtime import RuntimeStore, ScopedStateContext

_DENIED_ENV_PATTERNS = (*LLM_API_KEY_ENV_PATTERNS, "AZURE_OPENAI_*")
_MAX_COMMAND_ARGS = 64
_MAX_ARGUMENT_BYTES = 8 * 1024

ProcessOutputSink = Callable[["ProcessHandle", str], None | Awaitable[None]]
ProcessReaction = Callable[[str], None | Awaitable[None]]


class ProcessServiceError(RuntimeError):
    """A monitored process operation failed."""

    code = "process_service_error"


class ProcessCommandDeniedError(PermissionError, ProcessServiceError):
    """The command is not in the host allowlist."""

    code = "process_command_denied"


class ProcessNotFoundError(LookupError, ProcessServiceError):
    """The opaque process handle is unknown."""

    code = "process_not_found"


class ProcessInputError(ValueError, ProcessServiceError):
    """Process command or output limits are invalid."""

    code = "process_input_error"


@dataclass(frozen=True)
class ProcessFilter:
    """Bounded literal or regular-expression output filter."""

    pattern: str
    regex: bool = False

    def __post_init__(self) -> None:
        if not self.pattern or len(self.pattern.encode()) > 1024:
            raise ProcessInputError("Process output filter is outside the safe bounds")
        if self.regex:
            try:
                re.compile(self.pattern)
            except re.error as error:
                raise ProcessInputError("Process output filter is not valid regex") from error

    def matches(self, value: str) -> bool:
        return bool(re.search(self.pattern, value) if self.regex else self.pattern in value)


@dataclass(frozen=True)
class ProcessHandle:
    """Opaque process identity scoped to one LocalProcessService instance."""

    id: str


@dataclass(frozen=True)
class ProcessResult:
    """Bounded terminal process output."""

    handle: ProcessHandle
    status: str
    exit_code: int | None
    output: str
    truncated: bool = False
    error: str | None = None


@dataclass
class _RunningProcess:
    handle: ProcessHandle
    process: asyncio.subprocess.Process
    max_output_bytes: int
    output_filter: ProcessFilter | None
    output_sink: ProcessOutputSink | None
    output: bytearray
    truncated: bool = False
    status: str = "running"
    error: str | None = None
    exit_code: int | None = None
    task: asyncio.Task[None] | None = None
    done: asyncio.Event | None = None


class ProcessReactionRouter:
    """Route matching process output to an active Run or an idle follow-up callback."""

    def __init__(
        self,
        *,
        active: ProcessReaction | None = None,
        idle: ProcessReaction | None = None,
    ) -> None:
        self._active = active
        self._idle = idle

    def set_active(self, reaction: ProcessReaction | None) -> None:
        self._active = reaction

    def set_idle(self, reaction: ProcessReaction | None) -> None:
        self._idle = reaction

    async def __call__(self, handle: ProcessHandle, line: str) -> None:
        reaction = self._active or self._idle
        if reaction is None:
            return
        prompt = f"Monitored process {handle.id} output:\n{line.rstrip()}"
        result = reaction(prompt)
        if inspect.isawaitable(result):
            await result


class LocalProcessService:
    """Run bounded, allowlisted commands without a shell."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        allowed_commands: Sequence[str] | set[str],
        max_output_bytes: int = 64 * 1024,
        stop_timeout: float = 2.0,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            raise ProcessInputError("Process workspace must be a directory")
        allowed = frozenset(allowed_commands)
        if not allowed or any(not isinstance(command, str) or not command for command in allowed):
            raise ProcessInputError("Process command allowlist must not be empty")
        if isinstance(max_output_bytes, bool) or not 1 <= max_output_bytes <= 1024 * 1024:
            raise ProcessInputError("Process output limit is outside the safe range")
        if stop_timeout <= 0:
            raise ProcessInputError("Process stop timeout must be positive")
        self.workspace = root
        self.allowed_commands = allowed
        self.max_output_bytes = max_output_bytes
        self.stop_timeout = stop_timeout
        self.environment = _safe_environment(environment)
        self._processes: dict[str, _RunningProcess] = {}

    async def start(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        output_filter: ProcessFilter | None = None,
        output_sink: ProcessOutputSink | None = None,
    ) -> ProcessHandle:
        executable = Path(command).name
        if executable not in self.allowed_commands and command not in self.allowed_commands:
            raise ProcessCommandDeniedError(f"Command {executable!r} is not allowlisted")
        if not command or len(args) > _MAX_COMMAND_ARGS:
            raise ProcessInputError("Process command arguments are outside the safe bounds")
        if any(not isinstance(arg, str) or len(arg.encode()) > _MAX_ARGUMENT_BYTES for arg in args):
            raise ProcessInputError("Process argument is outside the safe bounds")
        try:
            process = await asyncio.create_subprocess_exec(
                command,
                *args,
                cwd=self.workspace,
                env=self.environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            raise ProcessServiceError("Allowlisted process could not start") from error
        handle = ProcessHandle(str(uuid4()))
        running = _RunningProcess(
            handle=handle,
            process=process,
            max_output_bytes=self.max_output_bytes,
            output_filter=output_filter,
            output_sink=output_sink,
            output=bytearray(),
            done=asyncio.Event(),
        )
        self._processes[handle.id] = running
        running.task = asyncio.create_task(self._read(running), name=f"yolop-process-{handle.id}")
        return handle

    async def wait(self, handle: ProcessHandle) -> ProcessResult:
        running = self._get(handle)
        assert running.done is not None
        try:
            await running.done.wait()
        except asyncio.CancelledError:
            await self.stop(handle)
            raise
        return self._result(running)

    def handles(self) -> tuple[ProcessHandle, ...]:
        return tuple(process.handle for process in self._processes.values())

    async def inspect(self, handle: ProcessHandle) -> ProcessResult:
        return self._result(self._get(handle))

    async def stop(self, handle: ProcessHandle) -> ProcessResult:
        running = self._get(handle)
        if running.status == "running":
            _signal_process_group(running.process, signal.SIGTERM)
            try:
                await asyncio.wait_for(running.process.wait(), timeout=self.stop_timeout)
            except TimeoutError:
                _signal_process_group(running.process, signal.SIGKILL)
                await running.process.wait()
            if running.task is not None:
                await asyncio.gather(running.task, return_exceptions=True)
        return self._result(running)

    async def aclose(self) -> None:
        handles = tuple(ProcessHandle(process_id) for process_id in self._processes)
        if handles:
            await asyncio.gather(*(self.stop(handle) for handle in handles))

    async def _read(self, running: _RunningProcess) -> None:
        assert running.process.stdout is not None
        try:
            while line := await running.process.stdout.readline():
                text = line.decode("utf-8", errors="replace")
                if running.output_filter is not None and not running.output_filter.matches(text):
                    continue
                if running.output_sink is not None:
                    try:
                        result = running.output_sink(running.handle, text)
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        running.error = "Process output reaction failed"
                        running.status = "failed"
                        if running.process.returncode is None:
                            _signal_process_group(running.process, signal.SIGTERM)
                        running.exit_code = await running.process.wait()
                        break
                if len(running.output) + len(line) > running.max_output_bytes:
                    overflow = len(running.output) + len(line) - running.max_output_bytes
                    del running.output[:overflow]
                    running.truncated = True
                running.output.extend(line[-running.max_output_bytes :])
            if running.process.returncode is None:
                running.exit_code = await running.process.wait()
            elif running.exit_code is None:
                running.exit_code = running.process.returncode
            if running.status == "running":
                running.status = "exited" if running.exit_code == 0 else "failed"
        except asyncio.CancelledError:
            if running.process.returncode is None:
                _signal_process_group(running.process, signal.SIGKILL)
                await running.process.wait()
            running.status = "cancelled"
            raise
        except Exception:
            running.status = "failed"
            running.exit_code = running.process.returncode
        finally:
            assert running.done is not None
            running.done.set()

    def _get(self, handle: ProcessHandle) -> _RunningProcess:
        try:
            return self._processes[handle.id]
        except KeyError as error:
            raise ProcessNotFoundError("Unknown process handle") from error

    @staticmethod
    def _result(running: _RunningProcess) -> ProcessResult:
        return ProcessResult(
            handle=running.handle,
            status=running.status,
            exit_code=running.exit_code,
            output=bytes(running.output).decode("utf-8", errors="replace"),
            truncated=running.truncated,
            error=running.error,
        )


@dataclass(frozen=True)
class ProcessMonitorRecord:
    """Durable monitor metadata; never a live process handle."""

    handle: ProcessHandle
    command: str
    status: str
    exit_code: int | None
    output: str
    updated_at: str


class ProcessMonitorStore:
    """Persist bounded monitor metadata in one Session-scoped RuntimeStore stream."""

    def __init__(
        self,
        runtime_store: RuntimeStore,
        *,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> None:
        self._state = ScopedStateContext(
            store=runtime_store,
            namespace=namespace,
            session_id=session_id,
            run_id=run_id,
        ).for_session("yolop.workspace.processes")

    async def list(self) -> tuple[ProcessMonitorRecord, ...]:
        entries = await self._state.read("monitors", schema_version=1)
        if not entries:
            return ()
        payload = entries[-1].payload
        if not isinstance(payload, dict) or not isinstance(payload.get("monitors"), dict):
            raise ProcessServiceError("Stored process monitor state is invalid")
        return tuple(
            _monitor_from_payload(value) for _, value in sorted(payload["monitors"].items())
        )

    async def get(self, handle: ProcessHandle) -> ProcessMonitorRecord | None:
        return next((record for record in await self.list() if record.handle == handle), None)

    async def save(self, record: ProcessMonitorRecord) -> None:
        entries = await self._state.read("monitors", schema_version=1)
        monitors: dict[str, dict[str, object]] = {}
        if entries:
            payload = entries[-1].payload
            if not isinstance(payload, dict) or not isinstance(payload.get("monitors"), dict):
                raise ProcessServiceError("Stored process monitor state is invalid")
            monitors = {
                str(handle): dict(value)
                for handle, value in payload["monitors"].items()
                if isinstance(value, dict)
            }
        monitors[record.handle.id] = _monitor_to_payload(record)
        expected = entries[-1].sequence if entries else 0
        await self._state.append(
            "monitors",
            {"monitors": monitors},
            schema_version=1,
            expected_sequence=expected,
        )

    async def reconcile(self) -> tuple[ProcessMonitorRecord, ...]:
        """Mark old running records unknown; never claim a process after restart."""
        records = await self.list()
        reconciled = tuple(
            ProcessMonitorRecord(
                handle=record.handle,
                command=record.command,
                status="unknown" if record.status == "running" else record.status,
                exit_code=record.exit_code,
                output=record.output,
                updated_at=record.updated_at,
            )
            for record in records
        )
        for old, record in zip(records, reconciled, strict=True):
            if old.status != record.status:
                await self.save(record)
        return reconciled


def _monitor_to_payload(record: ProcessMonitorRecord) -> dict[str, object]:
    return {
        "handle": record.handle.id,
        "command": record.command,
        "status": record.status,
        "exit_code": record.exit_code,
        "output": record.output,
        "updated_at": record.updated_at,
    }


def _monitor_from_payload(value: object) -> ProcessMonitorRecord:
    if not isinstance(value, dict):
        raise ProcessServiceError("Stored process monitor record is invalid")
    try:
        return ProcessMonitorRecord(
            handle=ProcessHandle(str(value["handle"])),
            command=str(value["command"]),
            status=str(value["status"]),
            exit_code=value.get("exit_code") if isinstance(value.get("exit_code"), int) else None,
            output=str(value.get("output", "")),
            updated_at=str(value["updated_at"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProcessServiceError("Stored process monitor record is invalid") from error


def _signal_process_group(process: asyncio.subprocess.Process, signum: int) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signum)
    except ProcessLookupError:
        return
    except PermissionError:
        process.send_signal(signum)


def _safe_environment(extra: Mapping[str, str] | None) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(fnmatch.fnmatchcase(key, pattern) for pattern in _DENIED_ENV_PATTERNS)
    }
    if extra is not None:
        if any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in extra.items()
        ):
            raise ProcessInputError("Process environment must contain strings")
        environment.update(extra)
    for key in tuple(environment):
        if any(fnmatch.fnmatchcase(key, pattern) for pattern in _DENIED_ENV_PATTERNS):
            environment.pop(key)
    return environment
