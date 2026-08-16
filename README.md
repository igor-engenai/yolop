# YoloP

YoloP is a small runtime for agents defined with Pydantic AI `AgentSpec`.

The core is stateless. Pydantic AI owns the model and tool loop. AgentSpec selects installed capability code. Hosts own AgentSpec files, trusted namespaces, dependencies, persistence, and transport.

## Install

Install only the core runtime:

```bash
uv add yolop
```

Install all YoloP hosts, capabilities, persistence packages, and OpenAI model support:

```bash
uv add "yolop[all]"
```

You can install smaller feature sets instead:

- `uv add "yolop[workspace]"` installs the Workspace capability.
- `uv add "yolop[duckdb]"` installs the DuckDB capability.
- `uv add "yolop[web]"` installs the web host and its runtime persistence.
- `uv add "yolop[tui,openai]"` installs the terminal host and OpenAI model support.
- `uv add "yolop[openai]"` installs OpenAI model support.

The feature packages remain separate Python distributions. A package index must contain compatible `0.1.x` releases of each selected YoloP distribution. For development in this repository, use `uv sync --all-packages`.

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

## Run the terminal host

Start the bundled Workspace coding AgentSpec:

```bash
export OPENAI_API_KEY="your-key"
uv run yolop
```

Use an external AgentSpec instead:

```bash
uv run yolop --agent-spec examples/agents/chat.yaml
```

The packaged command authorizes the current directory and injects it as the Workspace dependency. The bundled AgentSpec can read and write project files and run its explicit shell command allowlist. It uses `openai:gpt-5.6-luna` with high thinking and does not include Skills. An external AgentSpec fully replaces the bundled data and can select Workspace against the same host-authorized current directory. Other host resources still need an embedding host that injects them. Run the custom coding composition example with:

```bash
uv run python examples/coding_tui.py
```

The TUI is a true inline macOS/Linux terminal application. It keeps normal terminal scrollback and has one Pi-like multiline editor plus one status line. It stores exact Pydantic AI session history in project-local `.yolop/runtime.db`. A new session starts by default. Use `--session <id>` or `/resume` to continue one.

While a run streams, Enter sends a native `asap` steering message. Escape cancels the run and saves its partial history. The main keys are:

- Enter: submit or steer;
- Shift+Enter or Ctrl+J: insert a newline;
- Escape: cancel the active run;
- Ctrl+C: cancel an active run, clear non-empty input, or exit when idle and empty;
- Ctrl+D: exit when the editor is empty;
- Ctrl+O: show or hide tool details;
- Ctrl+T: show or hide thinking;
- PageUp/PageDown: scroll the transcript by one visible page;
- mouse wheel or trackpad: scroll transcript lines;
- End: return to the newest output and resume automatic following.

New streamed output follows automatically while the transcript is at the bottom. Scrolling up pauses that following until PageDown, mouse-down scrolling, or End reaches the bottom. Mouse mode is active while the TUI runs; use the terminal's modifier key, usually Shift, for native text selection.

The only built-in commands are `/new`, `/resume`, `/help`, and `/quit`. Type `@` to fuzzy-complete project files. Use `@path` or `@"path with spaces"` for an attachment. Quotes, apostrophes, and backticks elsewhere remain ordinary prompt text and are not parsed as shell syntax. Text attachments must stay inside the project, use UTF-8, contain no null bytes, remain below 256 KiB each, and remain below 1 MiB in total.

V1 does not include model switching, direct shell mode, images, an external editor, branching, remote-server mode, custom themes, custom keybindings, TUI extensions, or a native Windows support promise.

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

The host explicitly loads and saves the session around `Yolop.run()`. `WorkspaceRuntimeStore` stores namespaced sessions and runs in `.yolop/runtime.db`. A session pins the canonical AgentSpec digest and resolved model ID. A save uses the loaded revision and atomically replaces the exact Pydantic AI message history.

This is a breaking storage format. The runtime store rejects old 0.1 session databases and JSONL files. It does not migrate them.

## Query DuckDB with a session

```bash
uv run python examples/duckdb_agent.py \
  --database analytics.duckdb \
  "Show total revenue by month"
```

Resume the generated session against the same database:

```bash
uv run python examples/duckdb_agent.py \
  --database analytics.duckdb \
  --session 68a094a7-3953-4f89-9acf-d60dc343d529 \
  "Now group the result by customer segment"
```

The host opens the database with `read_only=True` and `enable_external_access=false`. The capability verifies both settings before it exposes the tool. It accepts one DuckDB SELECT-class statement per call. AgentSpec can set:

- `max_rows`, default `200`;
- `max_result_bytes`, default `1048576`;
- `timeout_seconds`, default `30`.

Timeout or cancellation interrupts DuckDB and waits for the worker to exit before the connection lock is released.

## Run the web chat

```bash
export OPENAI_API_KEY="your-key"
uv run python examples/web_chat.py
```

Open <http://127.0.0.1:8000>. The page creates a durable session and puts its ID in the URL. Reloading the URL restores the exact stored history.

The reusable command serves only the HTTP API:

```bash
uv run yolop-webserver --agent-spec examples/agents/chat.yaml
```

It binds to `127.0.0.1` and stores runtime state in `.yolop/runtime.db` by default. Select a workspace-local database explicitly:

```bash
uv run yolop-webserver \
  --agent-spec examples/agents/chat.yaml \
  --session-backend workspace \
  --session-path .
```

The API provides:

```text
POST   /v1/sessions
GET    /v1/sessions
GET    /v1/sessions/{session_id}
DELETE /v1/sessions/{session_id}
POST   /v1/sessions/{session_id}/runs
POST   /v1/sessions/{session_id}/runs/stream
```

Both run endpoints accept `{"prompt": "user text"}` and require an `Idempotency-Key` header. Reusing a key with the same prompt returns the same durable run. Reusing it with different input returns `409 idempotency_conflict`.

The JSON endpoint waits for durable completion. The SSE endpoint replays persisted native Pydantic AI events with sequence IDs, follows new events, and ends with `run_completed` or `run_error`. A client disconnect only detaches the stream. The accepted background run continues. Reconnect by sending the same POST and idempotency key.

Default admission limits are:

- 8 active model runs per process;
- 8 nonterminal runs per session;
- 64 supervised background runs per process;
- 30 seconds to acquire a session lock.

Hosts can change these values with `RunLimits`. Excess work fails with `429 run_queue_full`. A lock timeout fails with `503 session_lock_timeout`.

Authentication remains host policy. `create_app(...)` requires a trusted `namespace_resolver(Request)` and a per-session `deps_resolver(namespace, session_id)`. The client cannot choose the persistence namespace. An authenticated multi-user host must map its verified tenant identity to that resolver. Do not bind the example or bare CLI to a public interface.

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

YoloP returns Pydantic AI's native `AgentRunEvents` handle. It does not wrap events, results, messages, or dependencies. `Yolop.execute(...)` is the run-to-completion form for hosts that need Pydantic AI's native event handler and `RunContext`, including steering and cancellation.

## Packages

### `yolop`

The core package contains the stateless runtime, AgentSpec capability discovery, the generic Skills capability, and bundled default skills. It does not contain agent-specific AgentSpecs, Workspace code, DuckDB, persistence, FastAPI, model providers, or `pydantic-ai-harness`.

### `yolop-session`

[`packages/yolop-session`](packages/yolop-session) defines one storage-independent `RuntimeStore` protocol. It owns namespaced session and run values, execution pins, durable event values, stable errors, and generated identity helpers.

### `yolop-sqlite-session`

[`packages/yolop-sqlite-session`](packages/yolop-sqlite-session) implements `RuntimeStore` with SQLite. It provides atomic session/run completion, revision checks, idempotent run reservation, ordered events, worker leases, namespace isolation, and cross-worker file locks. It is the default web-server store.

### `yolop-webserver`

[`packages/yolop-webserver`](packages/yolop-webserver) is the FastAPI host. It provides request-scoped namespace and dependency resolution, JSON and SSE APIs, durable supervised runs, event replay, bounded admission, and a loopback CLI. The chat page remains under [`examples/`](examples/).

### `yolop-tui`

[`packages/yolop-tui`](packages/yolop-tui) is the minimal inline terminal host. It owns prompt_toolkit and Rich, the `yolop` command, a bundled Workspace coding AgentSpec, native steering and cancellation, compact tool and thinking views, `@file` policy, and local session selection. The packaged command depends on `yolop-workspace` and injects the current directory. The lower-level `run_tui(...)` still accepts caller-selected native dependencies. The package does not depend on DuckDB.

### `yolop-duckdb`

[`packages/yolop-duckdb`](packages/yolop-duckdb) is a separate capability plugin. It owns the DuckDB dependency, resolves a host-provided read-only connection, and exposes the bounded `query_duckdb` model tool.

### `yolop-workspace`

[`packages/yolop-workspace`](packages/yolop-workspace) owns the AgentSpec-selectable Workspace capability, `pydantic-ai-harness`, filesystem and shell composition, host workspace dependency requirements, and shell credential filtering. The shell command allowlist is explicit AgentSpec policy.

### `yolop-workspace-session`

[`packages/yolop-workspace-session`](packages/yolop-workspace-session) provides `WorkspaceRuntimeStore`. It applies workspace path policy to the shared SQLite runtime implementation and stores the database at `<workspace>/.yolop/runtime.db`. It is host persistence, not an AgentSpec capability.

## Agent configuration and code

AgentSpec contains declarative agent data: instructions, prompts, model configuration, capability names and arguments, bundled skills, and immutable inline custom skills.

Capability implementations and tools are installed Python code. YoloP resolves custom capability names through the `yolop.capabilities` entry-point group and imports only selected providers.

Changing prompts, inline skills, or capability arguments creates a new AgentSpec and needs no code release. Existing sessions keep their immutable AgentSpec digest and model pin. A host must create a new session or implement an explicit upgrade operation to change either pin.
