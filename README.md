# YoloP

YoloP is a small runtime for agents defined with Pydantic AI `AgentSpec`.

The kernel stays stateless. Pydantic AI owns the model and tool loop. AgentSpec selects installed capability code. Hosts own dependencies, history, persistence, and transport.

## Try the AgentSpec

```bash
uv sync
export OPENAI_API_KEY="your-key"
uv run python examples/agentspec.py
```

The example loads [`examples/agent.yaml`](examples/agent.yaml). AgentSpec selects `openai:gpt-5.6-luna`, sets `thinking: minimal`, and provides the agent description, prompt, and skills. YoloP streams native Pydantic AI events and reads the native result and message history.

The API key is not part of AgentSpec. Pydantic AI reads `OPENAI_API_KEY` from the environment. An Azure AgentSpec can use `azure:<deployment-name>` with `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`. Legacy Azure endpoints also need `OPENAI_API_VERSION`.

## Runtime API

```python
async with Yolop().run(
    spec,
    prompt,
    deps=deps,
    deps_type=type(deps),
    message_history=history,
) as run:
    async for event in run:
        handle(event)

result = run.result
messages = run.all_messages()
```

YoloP returns Pydantic AI's `AgentRunEvents` handle. It does not wrap events, results, messages, or dependencies.

## Agent configuration and code

AgentSpec contains declarative agent data:

- instructions and prompts;
- model configuration;
- capability names and arguments;
- selected bundled skills;
- immutable inline custom skill snapshots.

Capability implementations and tools are installed Python code. YoloP resolves custom capability names through the `yolop.capabilities` entry-point group and imports only selected providers. Changing prompts, inline skills, or capability arguments creates a new AgentSpec and needs no code release. Changing capability code or bundled skill files requires a package deployment.

Bundled skills are available but inactive by default. AgentSpec must explicitly enable them.
