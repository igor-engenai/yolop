from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from yolop_runtime import (
    CompactionUnsupportedError,
    ExecutionScope,
    Runtime,
    RuntimeSessionSnapshot,
)
from yolop_sqlite_session import SQLiteRuntimeStore


@dataclass
class FakeCompactor:
    focus: str | None = None
    calls: int = 0
    seen_scope: ExecutionScope | None = None

    async def compact(
        self,
        messages: Sequence[ModelMessage],
        *,
        focus: str | None,
        model: Any,
        deps: Any,
        scope: ExecutionScope,
    ) -> list[ModelMessage]:
        del model, deps
        self.calls += 1
        self.focus = focus
        self.seen_scope = scope
        return [*messages, ModelRequest(parts=[UserPromptPart(f"focused: {focus}")])]


async def _run_once(runtime: Runtime[Any], session: RuntimeSessionSnapshot) -> None:
    spec = AgentSpec(model="test:model")

    async def respond(_messages: list[ModelMessage], _info: AgentInfo):
        yield "before compaction"

    await runtime.run(
        "test",
        session.id,
        "initial prompt",
        spec=spec,
        model=FunctionModel(stream_function=respond),
        deps=None,
        deps_type=type(None),
        idempotency_key="initial",
    )


@pytest.mark.asyncio
async def test_manual_compaction_changes_active_history_only(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    spec = AgentSpec(model="test:model")
    session = await runtime.create_session("test", spec=spec, model_id="test:model")
    await _run_once(runtime, session)
    session = await runtime.load_session("test", session.id)
    run = (await runtime.list_runs("test", session_id=session.id))[0]
    compactor = FakeCompactor()

    compacted = await runtime.compact_session(
        "test",
        session.id,
        spec=spec,
        model="test:model",
        model_id="test:model",
        deps=None,
        compactor=compactor,
        focus="keep {braces}",
    )

    assert compactor.calls == 1
    assert compactor.focus == "keep {braces}"
    assert compactor.seen_scope is not None
    assert compacted.messages[-1] != run.full_messages[-1]
    stored_run = await runtime.get_run("test", run.id)
    assert stored_run.full_messages == run.full_messages


@pytest.mark.asyncio
async def test_manual_compaction_requires_selected_capability(tmp_path: Path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime.db")
    runtime = Runtime(store=store)
    spec = AgentSpec(model="test:model")
    session = await runtime.create_session("test", spec=spec, model_id="test:model")

    with pytest.raises(CompactionUnsupportedError, match="selected"):
        await runtime.compact_session(
            "test",
            session.id,
            spec=spec,
            model="test:model",
            model_id="test:model",
            deps=None,
            compactor=None,
        )
