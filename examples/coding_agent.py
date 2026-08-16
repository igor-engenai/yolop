import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from yolop import Yolop, coding_agent_spec


@dataclass(frozen=True)
class HostDeps:
    workspace: Path


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    spec = coding_agent_spec()
    deps = HostDeps(workspace=Path.cwd())

    async with Yolop().run(
        spec,
        "Inspect pyproject.toml and summarize what this package does. Do not change files.",
        deps=deps,
        deps_type=HostDeps,
    ) as run:
        async for _ in run:
            pass

    assert run.result is not None
    print(run.result.output)


if __name__ == "__main__":
    asyncio.run(main())
