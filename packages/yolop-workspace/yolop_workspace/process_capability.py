from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic_ai.capabilities import Capability
from pydantic_ai.tools import RunContext

from .processes import (
    LocalProcessService,
    ProcessFilter,
    ProcessHandle,
    ProcessMonitorRecord,
    ProcessMonitorStore,
    ProcessOutputSink,
    ProcessResult,
)


def build_process_capability(
    service: LocalProcessService,
    *,
    monitor_store: ProcessMonitorStore | None = None,
    output_sink: ProcessOutputSink | None = None,
) -> Capability[Any]:
    """Build native process tools around one host-owned service."""
    capability = Capability[Any](
        id="yolop.processes",
        description="Start and inspect explicitly allowlisted workspace processes.",
    )

    @capability.tool
    async def process_start(
        _ctx: RunContext[Any],
        command: str,
        args: list[str] | None = None,
        filter_pattern: str | None = None,
        filter_regex: bool = False,
    ) -> dict[str, Any]:
        output_filter = (
            ProcessFilter(filter_pattern, regex=filter_regex)
            if filter_pattern is not None
            else None
        )
        handle = await service.start(
            command,
            () if args is None else args,
            output_filter=output_filter,
            output_sink=output_sink,
        )
        if monitor_store is not None:
            await monitor_store.save(
                ProcessMonitorRecord(
                    handle=handle,
                    command=command,
                    status="running",
                    exit_code=None,
                    output="",
                    updated_at=_now(),
                )
            )
        return {"handle": handle.id, "status": "running"}

    @capability.tool
    async def process_list(_ctx: RunContext[Any]) -> list[dict[str, Any]]:
        if monitor_store is not None:
            return [_monitor_payload(record) for record in await monitor_store.list()]
        return [_result_payload(await service.inspect(handle)) for handle in service.handles()]

    @capability.tool
    async def process_inspect(_ctx: RunContext[Any], handle: str) -> dict[str, Any]:
        result = await service.inspect(ProcessHandle(handle))
        await _save_result(monitor_store, result, command="unknown")
        return _result_payload(result)

    @capability.tool
    async def process_wait(_ctx: RunContext[Any], handle: str) -> dict[str, Any]:
        result = await service.wait(ProcessHandle(handle))
        await _save_result(monitor_store, result, command="unknown")
        return _result_payload(result)

    @capability.tool
    async def process_stop(_ctx: RunContext[Any], handle: str) -> dict[str, Any]:
        result = await service.stop(ProcessHandle(handle))
        await _save_result(monitor_store, result, command="unknown")
        return _result_payload(result)

    return capability


async def _save_result(
    monitor_store: ProcessMonitorStore | None,
    result: ProcessResult,
    *,
    command: str,
) -> None:
    if monitor_store is None:
        return
    existing = await monitor_store.get(result.handle)
    await monitor_store.save(
        ProcessMonitorRecord(
            handle=result.handle,
            command=existing.command if existing is not None else command,
            status=result.status,
            exit_code=result.exit_code,
            output=result.output,
            updated_at=_now(),
        )
    )


def _result_payload(result: ProcessResult) -> dict[str, Any]:
    return {
        "handle": result.handle.id,
        "status": result.status,
        "exit_code": result.exit_code,
        "output": result.output,
        "truncated": result.truncated,
        "error": result.error,
    }


def _monitor_payload(record: ProcessMonitorRecord) -> dict[str, Any]:
    return {
        "handle": record.handle.id,
        "command": record.command,
        "status": record.status,
        "exit_code": record.exit_code,
        "output": record.output,
        "updated_at": record.updated_at,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()
