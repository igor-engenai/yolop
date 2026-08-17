import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pydantic_ai import AgentSpec
from yolop_runtime import ExecutionPin
from yolop_sqlite_session import SQLiteRuntimeStore

from .app import run_tui

_DEFAULT_SPEC = Path(__file__).with_name("agent_specs") / "coding.yaml"


@dataclass(frozen=True)
class _HostDeps:
    workspace: Path


def main(argv: Sequence[str] | None = None) -> None:
    """Run one AgentSpec in the local Textual terminal host."""
    args = _parser().parse_args(argv)
    spec = AgentSpec.from_file(args.agent_spec or _DEFAULT_SPEC)
    if not isinstance(spec.model, str):
        raise SystemExit("The TUI AgentSpec must contain a string model reference")
    pin = ExecutionPin.from_spec(spec, model_id=spec.model)
    namespace = _execution_namespace(pin)
    store = SQLiteRuntimeStore(args.database)
    workspace = Path.cwd().resolve()
    deps = _HostDeps(workspace=workspace)
    asyncio.run(
        run_tui(
            spec,
            store=store,
            namespace=namespace,
            deps=deps,
            deps_type=_HostDeps,
            session_id=args.session,
            cwd=workspace,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a YoloP agent in the Textual terminal host.")
    parser.add_argument(
        "--agent-spec",
        type=Path,
        help="AgentSpec YAML file; defaults to the bundled Workspace coding agent",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(".yolop/runtime.db"),
        help="runtime SQLite path (default: .yolop/runtime.db)",
    )
    parser.add_argument("--session", help="resume a generated session ID")
    return parser


def _execution_namespace(pin: ExecutionPin) -> str:
    digest = sha256(f"{pin.agent_spec_id}\0{pin.model_id}".encode()).hexdigest()
    return f"tui:{digest}"


if __name__ == "__main__":
    main()
