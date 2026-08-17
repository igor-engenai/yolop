from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext


@dataclass
class RewriteRequestHistory(AbstractCapability[None]):
    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        del ctx
        request_context.messages[:] = [
            ModelRequest(parts=[UserPromptPart("rewritten active request")])
        ]
        return request_context


def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("answer")])


async def test_result_history_semantics_after_capability_rewrites_request_history() -> None:
    history = [ModelRequest(parts=[UserPromptPart("canonical prior history")])]
    result = await Agent[None, str](
        FunctionModel(function=respond),
        capabilities=[RewriteRequestHistory()],
    ).run(
        "current prompt",
        message_history=history,
    )

    new_messages = result.new_messages()
    all_messages = result.all_messages()

    assert new_messages == all_messages
    assert isinstance(new_messages[0], ModelRequest)
    assert isinstance(new_messages[0].parts[0], UserPromptPart)
    assert new_messages[0].parts[0].content == "rewritten active request"
    assert isinstance(new_messages[-1], ModelResponse)
    assert new_messages[-1].parts == [TextPart("answer")]
    assert history not in all_messages
