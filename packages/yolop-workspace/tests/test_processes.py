import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic_ai import AgentRunResultEvent, AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pytest import raises
from yolop_runtime import ExecutionPin
from yolop_sqlite_session import SQLiteRuntimeStore
from yolop_workspace import (
    LocalProcessService,
    ProcessCommandDeniedError,
    ProcessFilter,
    ProcessHandle,
    ProcessMonitorRecord,
    ProcessMonitorStore,
    ProcessReactionRouter,
    build_process_capability,
)

from yolop import Yolop


async def test_process_monitor_state_reconciles_stale_running_records(tmp_path) -> None:
    database = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(database)
    session = await store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="test:model"),
    )
    monitor_store = ProcessMonitorStore(
        store,
        namespace="tenant/acme",
        session_id=session.id,
        run_id=session.id,
    )
    await monitor_store.save(
        ProcessMonitorRecord(
            handle=ProcessHandle("opaque"),
            command="python",
            status="running",
            exit_code=None,
            output="ready",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    reopened = ProcessMonitorStore(
        SQLiteRuntimeStore(database),
        namespace="tenant/acme",
        session_id=session.id,
        run_id=session.id,
    )
    records = await reopened.reconcile()

    assert records[0].status == "unknown"
    assert (await reopened.list())[0].status == "unknown"


async def test_process_reactions_prefer_active_runs_and_fall_back_to_idle_runs() -> None:
    active: list[str] = []
    idle: list[str] = []
    router = ProcessReactionRouter(active=active.append, idle=idle.append)

    await router(ProcessHandle("process"), "ready\n")
    router.set_active(None)
    await router(ProcessHandle("process"), "failed\n")

    assert active == ["Monitored process process output:\nready"]
    assert idle == ["Monitored process process output:\nfailed"]


async def test_local_process_service_runs_an_allowlisted_command(tmp_path) -> None:
    service = LocalProcessService(
        tmp_path,
        allowed_commands={Path(sys.executable).name},
    )
    handle = await service.start(
        sys.executable,
        ["-c", "print('ready', flush=True)"],
    )

    result = await service.wait(handle)

    assert result.status == "exited"
    assert result.exit_code == 0
    assert "ready" in result.output
    await service.aclose()


async def test_native_process_capability_starts_a_process(tmp_path) -> None:
    service = LocalProcessService(
        tmp_path,
        allowed_commands={Path(sys.executable).name},
    )
    capability = build_process_capability(service)

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            assert {
                "process_start",
                "process_list",
                "process_inspect",
                "process_wait",
                "process_stop",
            } <= {tool.name for tool in info.function_tools}
            yield {
                0: DeltaToolCall(
                    name="process_start",
                    json_args=json.dumps(
                        {
                            "command": sys.executable,
                            "args": ["-c", "print('started')"],
                        }
                    ),
                    tool_call_id="process-start",
                )
            }
        else:
            yield "started"

    async with Yolop().run(
        AgentSpec(),
        "start a process",
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        mandatory_capabilities=[capability],
    ) as run:
        events = [event async for event in run]
    result = await service.wait(service.handles()[0])

    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "started"
    assert result.output == "started\n"
    await service.aclose()


async def test_local_process_service_cancelling_wait_stops_the_process(tmp_path) -> None:
    service = LocalProcessService(
        tmp_path,
        allowed_commands={Path(sys.executable).name},
    )
    handle = await service.start(sys.executable, ["-c", "import time; time.sleep(30)"])
    waiting = asyncio.create_task(service.wait(handle))
    await asyncio.sleep(0.05)

    waiting.cancel()
    with raises(asyncio.CancelledError):
        await waiting

    assert (await service.inspect(handle)).status != "running"
    await service.aclose()


async def test_local_process_service_filters_and_bounds_output(tmp_path) -> None:
    service = LocalProcessService(
        tmp_path,
        allowed_commands={Path(sys.executable).name},
        max_output_bytes=8,
    )
    handle = await service.start(
        sys.executable,
        ["-c", "print('skip'); print('match')"],
        output_filter=ProcessFilter("match"),
    )

    result = await service.wait(handle)

    assert result.output == "match\n"
    assert result.truncated is False
    await service.aclose()


async def test_local_process_service_bounds_unfiltered_output(tmp_path) -> None:
    service = LocalProcessService(
        tmp_path,
        allowed_commands={Path(sys.executable).name},
        max_output_bytes=8,
    )
    handle = await service.start(sys.executable, ["-c", "print('0123456789')"])

    result = await service.wait(handle)

    assert len(result.output.encode()) <= 8
    assert result.truncated is True
    await service.aclose()


async def test_local_process_service_does_not_inherit_model_credentials(tmp_path) -> None:
    service = LocalProcessService(
        tmp_path,
        allowed_commands={Path(sys.executable).name},
        environment={"OPENAI_API_KEY": "secret", "SAFE_PROCESS_VALUE": "yes"},
    )
    handle = await service.start(
        sys.executable,
        [
            "-c",
            "import os; print(os.getenv('OPENAI_API_KEY')); print(os.getenv('SAFE_PROCESS_VALUE'))",
        ],
    )

    result = await service.wait(handle)

    assert result.output == "None\nyes\n"
    await service.aclose()


async def test_local_process_service_delivers_matching_output_once(tmp_path) -> None:
    service = LocalProcessService(
        tmp_path,
        allowed_commands={Path(sys.executable).name},
    )
    seen: list[str] = []

    async def on_output(_handle, line: str) -> None:
        seen.append(line)

    handle = await service.start(
        sys.executable,
        ["-c", "print('ready', flush=True)"],
        output_sink=on_output,
    )
    await service.wait(handle)

    assert seen == ["ready\n"]
    await service.aclose()


async def test_local_process_service_rejects_a_forbidden_command(tmp_path) -> None:
    service = LocalProcessService(tmp_path, allowed_commands={"python"})

    with raises(ProcessCommandDeniedError):
        await service.start("sh", ["-c", "echo should-not-run"])

    await service.aclose()
