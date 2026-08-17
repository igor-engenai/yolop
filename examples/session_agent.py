import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import AgentSpec
from yolop_runtime import ExecutionPin, ensure_session_pin
from yolop_workspace_session import WorkspaceRuntimeStore

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
    store = WorkspaceRuntimeStore(workspace)
    spec = AgentSpec.from_file(SPEC_PATH)
    assert isinstance(spec.model, str)
    pin = ExecutionPin.from_spec(spec, model_id=spec.model)
    session = (
        await store.load_session("local", args.session)
        if args.session
        else await store.create_session("local", pin=pin)
    )
    ensure_session_pin(session, pin)
    print(f"session: {session.id}")

    deps = HostDeps(workspace=workspace)
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
    await store.replace_session(
        "local",
        session.id,
        expected_revision=session.revision,
        messages=run.all_messages(),
    )
    print(run.result.output)


if __name__ == "__main__":
    asyncio.run(main())
