import asyncio
import os
from pathlib import Path

from pydantic_ai import AgentSpec

from yolop import Yolop

SPEC_PATH = Path(__file__).with_name("agent.yaml")


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    spec = AgentSpec.from_file(SPEC_PATH)

    async with Yolop().run(
        spec,
        (
            "Load both skills, then explain how you would make one safe Python change. "
            "And count to 100"
        ),
        deps=None,
        deps_type=type(None),
    ) as run:
        async for event in run:
            print(type(event).__name__)

    assert run.result is not None
    print(f"result: {run.result.output}")
    print(f"messages: {len(run.all_messages())}")


if __name__ == "__main__":
    asyncio.run(main())
