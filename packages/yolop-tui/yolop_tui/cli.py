import argparse
import asyncio
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from pydantic_ai import AgentSpec
from yolop_session import ExecutionPin
from yolop_sqlite_session import SQLiteRuntimeStore

from .app import run_tui

_DEFAULT_SPEC = Path(__file__).with_name("agent_specs") / "chat.yaml"


def main(argv: Sequence[str] | None = None) -> None:
    """Run one AgentSpec in the local inline terminal host."""
    args = _parser().parse_args(argv)
    spec = AgentSpec.from_file(args.agent_spec or _DEFAULT_SPEC)
    if not isinstance(spec.model, str):
        raise SystemExit("The TUI AgentSpec must contain a string model reference")
    pin = ExecutionPin.from_spec(spec, model_id=spec.model)
    namespace = _execution_namespace(pin)
    store = SQLiteRuntimeStore(args.database)
    asyncio.run(
        run_tui(
            spec,
            store=store,
            namespace=namespace,
            deps=None,
            deps_type=type(None),
            session_id=args.session,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-spec", type=Path)
    parser.add_argument("--database", type=Path, default=Path(".yolop/runtime.db"))
    parser.add_argument("--session")
    return parser


def _execution_namespace(pin: ExecutionPin) -> str:
    digest = sha256(f"{pin.agent_spec_id}\0{pin.model_id}".encode()).hexdigest()
    return f"tui:{digest}"


if __name__ == "__main__":
    main()
