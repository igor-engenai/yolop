from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

from pydantic_ai import Agent, AgentRunEvents, AgentRunResult, AgentSpec
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import KnownModelName, Model

from .capabilities import load_capability_types


class Yolop:
    """Construct and run a fresh Pydantic AI Agent for each invocation."""

    async def execute[DepsT](
        self,
        spec: AgentSpec | dict[str, Any],
        prompt: str | Sequence[UserContent] | None,
        *,
        event_stream_handler: EventStreamHandler[DepsT],
        deps: DepsT,
        deps_type: type[DepsT],
        model: Model | KnownModelName | str | None = None,
        message_history: Sequence[ModelMessage] | None = None,
    ) -> AgentRunResult[Any]:
        """Run to completion with a native handler that can steer through RunContext."""
        capability_types = load_capability_types(spec)
        agent = Agent.from_spec(
            spec,
            deps_type=deps_type,
            model=model,
            custom_capability_types=capability_types,
        )
        return await agent.run(
            prompt,
            deps=deps,
            message_history=message_history,
            event_stream_handler=event_stream_handler,
        )

    def run[DepsT](
        self,
        spec: AgentSpec | dict[str, Any],
        prompt: str | Sequence[UserContent] | None,
        *,
        deps: DepsT,
        deps_type: type[DepsT],
        model: Model | KnownModelName | str | None = None,
        message_history: Sequence[ModelMessage] | None = None,
    ) -> AbstractAsyncContextManager[AgentRunEvents[Any]]:
        capability_types = load_capability_types(spec)
        agent = Agent.from_spec(
            spec,
            deps_type=deps_type,
            model=model,
            custom_capability_types=capability_types,
        )
        return agent.run_stream_events(prompt, deps=deps, message_history=message_history)
