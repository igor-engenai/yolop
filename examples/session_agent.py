import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import AgentSpec
from yolop_workspace_session import WorkspaceSessionStore

from yolop import Yolop

SPEC_PATH = Path(__file__).with_name("agents") / "coding.yaml"


@dataclass(frozen=True)
class HostDeps:
    workspace: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--session", help="Resume a generated session ID")
    return parser.parse_args()


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    args = parse_args()
    workspace = Path.cwd()
    store = WorkspaceSessionStore(workspace)
    session = await store.load(args.session) if args.session else await store.create()
    print(f"session: {session.id}")

    deps = HostDeps(workspace=workspace)
    spec = AgentSpec.from_file(SPEC_PATH)
    async with Yolop().run(
        spec,
        args.prompt,
        deps=deps,
        deps_type=HostDeps,
        message_history=session.messages,
    ) as run:
        async for _ in run:
            pass

    assert run.result is not None
    await store.replace(
        session.id,
        expected_revision=session.revision,
        messages=run.all_messages(),
    )
    print(run.result.output)


if __name__ == "__main__":
    asyncio.run(main())
