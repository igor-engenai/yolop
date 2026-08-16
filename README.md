# YoloP

YoloP is a small runtime for agents defined with Pydantic AI `AgentSpec`.

The core is stateless. Pydantic AI owns the model and tool loop. AgentSpec selects installed capability code. Hosts own AgentSpec files, dependencies, history, persistence, and transport.

## Try the coding AgentSpec

```bash
uv sync
export OPENAI_API_KEY="your-key"
uv run python examples/coding_agent.py
```

The example loads [`examples/agents/coding.yaml`](examples/agents/coding.yaml) directly with Pydantic AI. The AgentSpec lives outside the YoloP core package. It contains the model, description, instructions, model settings, and capability configuration.

The coding AgentSpec selects:

- `Workspace` file tools;
- an allowlisted workspace shell;
- the bundled `tdd` skill.

The host supplies the workspace path through dependencies. AgentSpec cannot select an arbitrary host directory. Shell subprocesses do not receive common LLM API keys, including OpenAI and Azure OpenAI keys.

The API key is not part of AgentSpec. Pydantic AI reads `OPENAI_API_KEY` from the environment. An Azure AgentSpec can use `azure:<deployment-name>` with `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`. Legacy Azure endpoints also need `OPENAI_API_VERSION`.

## Use workspace sessions

Create a session:

```bash
uv run python examples/session_agent.py "Inspect this project"
# session: 8c02a928-79b0-46de-8c38-11a5d416e1ca
```

Resume it with the generated ID:

```bash
uv run python examples/session_agent.py \
  --session 8c02a928-79b0-46de-8c38-11a5d416e1ca \
  "Continue the previous task"
```

The host explicitly loads and saves the session around `Yolop.run()`. Each session is stored at `.yolop/sessions/<session-id>.jsonl` in the current workspace. A save passes the loaded revision and atomically replaces the full history with `run.all_messages()`.

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

## Packages

### `yolop`

The core package contains:

- the stateless runtime;
- AgentSpec-based capability discovery;
- the generic Skills capability and bundled default skills.

It does not contain agent-specific AgentSpec files, Workspace code, model providers, or `pydantic-ai-harness`.

### `yolop-workspace`

[`packages/yolop-workspace`](packages/yolop-workspace) is a separate first-party capability plugin. It owns:

- the `Workspace` capability;
- the `pydantic-ai-harness` dependency;
- filesystem and shell composition;
- host workspace dependency requirements;
- shell credential filtering.

The shell command allowlist is agent policy. It is explicit in the external coding AgentSpec, not a Python default.

### `yolop-workspace-session`

[`packages/yolop-workspace-session`](packages/yolop-workspace-session) is a separate host persistence package. It provides multiple generated sessions with native Pydantic AI message JSONL, atomic full-history replacement, revision checks, and per-session write locks.

It is not an AgentSpec capability. Hosts call it explicitly, and YoloP core does not depend on it.

## Agent configuration and code

AgentSpec contains declarative agent data:

- instructions and prompts;
- model configuration;
- capability names and arguments;
- selected bundled skills;
- immutable inline custom skill snapshots.

Capability implementations and tools are installed Python code. YoloP resolves custom capability names through the `yolop.capabilities` entry-point group and imports only selected providers.

The coding and research examples in [`pydantic-ai-harness`](https://github.com/pydantic/pydantic-ai-harness/tree/v0.21.0/examples) are composition references. YoloP uses harness code through optional capability plugins, but agents remain external AgentSpec data. It does not copy the Python `build_agent()` factory pattern.

Changing prompts, inline skills, or capability arguments creates a new AgentSpec and needs no code release. Changing capability code or bundled skill files requires a package deployment. Bundled skills are available but inactive until AgentSpec enables them.
