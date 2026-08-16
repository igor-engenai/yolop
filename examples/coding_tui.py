import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import AgentSpec
from yolop_tui import run_tui
from yolop_workspace_session import WorkspaceRuntimeStore

SPEC_PATH = Path(__file__).with_name("agents") / "coding.yaml"


@dataclass(frozen=True)
class HostDeps:
    workspace: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", help="Resume a generated session ID")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    workspace = Path.cwd().resolve()
    spec = AgentSpec.from_file(SPEC_PATH)
    await run_tui(
        spec,
        store=WorkspaceRuntimeStore(workspace),
        namespace="coding",
        deps=HostDeps(workspace=workspace),
        deps_type=HostDeps,
        session_id=args.session,
        cwd=workspace,
    )


if __name__ == "__main__":
    asyncio.run(main())
