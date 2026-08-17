from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

from pydantic_ai import Agent, AgentRunEvents, AgentRunResult, AgentSpec
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.usage import UsageLimits

from .capabilities import CapabilityResolution, resolve_capabilities
from .catalog import ProviderCatalog
from .providers import resolve_model_reference


class Yolop:
    """Construct and run a fresh Pydantic AI Agent for each invocation."""

    def __init__(self, provider_catalog: ProviderCatalog | None = None) -> None:
        self.provider_catalog = (
            ProviderCatalog.from_installed() if provider_catalog is None else provider_catalog
        )

    def resolve_capabilities[DepsT](
        self,
        spec: AgentSpec | dict[str, Any],
        *,
        mandatory_capabilities: Sequence[AbstractCapability[DepsT]] = (),
    ) -> CapabilityResolution:
        """Resolve AgentSpec capabilities and host-enforced policy separately."""
        return resolve_capabilities(
            spec,
            catalog=self.provider_catalog,
            mandatory_capabilities=mandatory_capabilities,
        )

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
        usage_limits: UsageLimits | None = None,
        mandatory_capabilities: Sequence[AbstractCapability[DepsT]] = (),
    ) -> AgentRunResult[Any]:
        """Run to completion with a native handler that can steer through RunContext."""
        capabilities = self.resolve_capabilities(
            spec,
            mandatory_capabilities=mandatory_capabilities,
        )
        resolved_model = _resolve_model(spec, model, catalog=self.provider_catalog)
        agent = Agent.from_spec(
            spec,
            deps_type=deps_type,
            model=resolved_model,
            custom_capability_types=capabilities.selected_types,
            capabilities=capabilities.enforced_capabilities,
        )
        return await agent.run(
            prompt,
            deps=deps,
            message_history=message_history,
            usage_limits=usage_limits,
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
        mandatory_capabilities: Sequence[AbstractCapability[DepsT]] = (),
    ) -> AbstractAsyncContextManager[AgentRunEvents[Any]]:
        capabilities = self.resolve_capabilities(
            spec,
            mandatory_capabilities=mandatory_capabilities,
        )
        resolved_model = _resolve_model(spec, model, catalog=self.provider_catalog)
        agent = Agent.from_spec(
            spec,
            deps_type=deps_type,
            model=resolved_model,
            custom_capability_types=capabilities.selected_types,
            capabilities=capabilities.enforced_capabilities,
        )
        return agent.run_stream_events(prompt, deps=deps, message_history=message_history)


def _resolve_model(
    spec: AgentSpec | dict[str, Any],
    model: Model | KnownModelName | str | None,
    *,
    catalog: ProviderCatalog,
) -> Model | KnownModelName | str | None:
    if isinstance(model, str):
        return resolve_model_reference(model, catalog=catalog)
    if model is not None:
        return model
    validated_spec = spec if isinstance(spec, AgentSpec) else AgentSpec.model_validate(spec)
    if isinstance(validated_spec.model, str):
        return resolve_model_reference(validated_spec.model, catalog=catalog)
    return None
