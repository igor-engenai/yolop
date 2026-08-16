# YoloP

YoloP is a small runtime for agents defined with Pydantic AI `AgentSpec`.

The kernel is stateless. Pydantic AI owns the model and tool loop. AgentSpec selects installed capability code. Hosts own dependencies, history, persistence, and transport.

## Try the coding AgentSpec

```bash
uv sync
export OPENAI_API_KEY="your-key"
uv run python examples/coding_agent.py
```

The example loads [`yolop/agent_specs/coding.yaml`](yolop/agent_specs/coding.yaml). This file is an exact Pydantic AI `AgentSpec`. It contains the model, description, instructions, model settings, and capability configuration.

The coding AgentSpec enables:

- `Workspace` file tools;
- an allowlisted workspace shell;
- the bundled `tdd` skill.

The host supplies the workspace path through dependencies. AgentSpec cannot select an arbitrary host directory. Shell subprocesses do not receive common LLM API keys, including OpenAI and Azure OpenAI keys.

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

YoloP returns Pydantic AI's native `AgentRunEvents` handle. It does not wrap events, results, messages, or dependencies.

## Agent configuration and code

AgentSpec contains declarative agent data:

- instructions and prompts;
- model configuration;
- capability names and arguments;
- selected bundled skills;
- immutable inline custom skill snapshots.

Capability implementations and tools are installed Python code. YoloP resolves custom capability names through the `yolop.capabilities` entry-point group and imports only selected providers.

The coding and research examples in [`pydantic-ai-harness`](https://github.com/pydantic/pydantic-ai-harness/tree/v0.21.0/examples) are composition references. YoloP reuses harness capabilities such as `FileSystem` and `Shell`, but YoloP agents remain AgentSpec data. It does not copy the Python `build_agent()` factory pattern.

Changing prompts, inline skills, or capability arguments creates a new AgentSpec and needs no code release. Changing capability code or bundled skill files requires a package deployment. Bundled skills are available but inactive until AgentSpec enables them.
