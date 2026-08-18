from __future__ import annotations

from pathlib import Path

from pytest import raises
from test_forks import setup
from yolop_deep import CandidateWorkspaceService, WorkspaceIsolationError
from yolop_runtime import ExecutionPin, Runtime
from yolop_sqlite_session import SQLiteRuntimeStore


async def test_candidate_workspace_is_a_copy_and_does_not_mutate_source(tmp_path: Path) -> None:
    runtime, source_session_id, source_run_id = await setup(tmp_path / "runtime")
    source = tmp_path / "workspace"
    source.mkdir()
    (source / "README.md").write_text("parent", encoding="utf-8")
    service = CandidateWorkspaceService(runtime, root=tmp_path)
    handle = await service.allocate(
        "tenant/acme",
        source_session_id,
        source_run_id,
        candidate_session_id="00000000-0000-4000-8000-000000000001",
        candidate_key="candidate-1",
        source_workspace=source,
    )

    (handle.path / "README.md").write_text("candidate", encoding="utf-8")

    assert (source / "README.md").read_text(encoding="utf-8") == "parent"
    assert handle.path != source
    assert handle.path.exists()
    assert (handle.path / "README.md").read_text(encoding="utf-8") == "candidate"


async def test_candidate_workspace_persists_reconnects_and_cleans_up(tmp_path: Path) -> None:
    runtime = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    source_session = await runtime.store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="model"),
    )
    source = tmp_path / "workspace"
    source.mkdir()
    service = CandidateWorkspaceService(runtime, root=tmp_path)
    first = await service.allocate(
        "tenant/acme",
        source_session.id,
        source_session.id,
        candidate_session_id="00000000-0000-4000-8000-000000000002",
        candidate_key="candidate-2",
        source_workspace=source,
    )
    restarted = CandidateWorkspaceService(runtime, root=tmp_path)
    second = await restarted.allocate(
        "tenant/acme",
        source_session.id,
        source_session.id,
        candidate_session_id="00000000-0000-4000-8000-000000000002",
        candidate_key="candidate-2",
        source_workspace=source,
    )

    assert second == first
    await restarted.cleanup(first)
    assert not first.path.exists()
    assert (
        await restarted.list_workspaces("tenant/acme", source_session.id, source_session.id) == []
    )


async def test_candidate_workspace_rejects_symlink_source_and_outside_root(tmp_path: Path) -> None:
    runtime = Runtime(store=SQLiteRuntimeStore(tmp_path / "runtime.db"))
    source_session = await runtime.store.create_session(
        "tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="model"),
    )
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (source / "link").symlink_to(outside)
    service = CandidateWorkspaceService(runtime, root=root)

    with raises(WorkspaceIsolationError):
        await service.allocate(
            "tenant/acme",
            source_session.id,
            source_session.id,
            candidate_session_id="00000000-0000-4000-8000-000000000003",
            candidate_key="candidate-3",
            source_workspace=source,
        )
    with raises(WorkspaceIsolationError):
        await service.allocate(
            "tenant/acme",
            source_session.id,
            source_session.id,
            candidate_session_id="00000000-0000-4000-8000-000000000004",
            candidate_key="candidate-4",
            source_workspace=tmp_path,
        )
