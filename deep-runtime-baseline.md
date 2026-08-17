# Deep-runtime baseline

Captured for `YDA-T00` at commit `e73f33b` on 2026-08-17.

This document records the public package surface before deep-runtime work. The T00 change adds only distribution-boundary tests. It does not change runtime behavior.

## Product boundary

- `yolop` is the small core runtime around Pydantic AI `AgentSpec`.
- AgentSpec capability providers are discovered through the `yolop.capabilities` entry-point group.
- Host applications and optional capabilities are separate distributions.
- The core package's non-extra runtime dependencies are exactly:
  - `pydantic-ai-slim[spec]>=2.31.0`
  - `pyyaml>=6.0.3`
- Textual, FastAPI, SQLite, `pydantic-ai-harness`, MCP, DuckDB, and optional model providers are not core runtime dependencies.

## Distributions

The workspace contains these version `0.1.0` distributions:

- `yolop`
- `yolop-duckdb`
- `yolop-providers`
- `yolop-session`
- `yolop-sqlite-session`
- `yolop-tui`
- `yolop-webserver`
- `yolop-workspace`
- `yolop-workspace-session`

## Entry points

| Group | Name | Target |
| --- | --- | --- |
| `yolop.capabilities` | `Skills` | `yolop.skills:Skills` |
| `yolop.capabilities` | `DuckDB` | `yolop_duckdb:DuckDB` |
| `yolop.capabilities` | `Workspace` | `yolop_workspace:Workspace` |
| `yolop.model_providers` | `openai-codex` | `yolop_providers:create_codex_model` |
| `yolop.auth_providers` | `openai-codex` | `yolop_providers:CodexOAuth` |

The installed console commands are `yolop`, `yolop-providers`, and `yolop-webserver`.

## Supported public imports

- `yolop.Yolop`
- `yolop_duckdb.DuckDB`, `yolop_duckdb.DuckDBDeps`
- `yolop_providers` public provider and credential classes
- `yolop_session` runtime store contracts and session/run types
- `yolop_sqlite_session.SQLiteRuntimeStore`
- `yolop_tui.run_tui`
- `yolop_webserver.create_app`, `yolop_webserver.RunLimits`
- `yolop_workspace.Workspace`, `yolop_workspace.WorkspaceDeps`
- `yolop_workspace_session.WorkspaceRuntimeStore`

## Wheel smoke

`uv build --all-packages` built all nine wheels and source distributions. The nine wheels were extracted into an isolated directory. All nine public modules imported, and every installed YoloP entry point loaded successfully.

## Test baseline

The complete suite passed before the T00 test additions: **176 passed**.

Collected focused coverage:

| Area | Tests |
| --- | ---: |
| Core capability loading | 4 |
| TUI | 67 |
| Web server | 20 |
| Runtime store contracts and SQLite store | 22 |
| Providers | 27 |
| Workspace capability | 3 |
| Workspace session | 1 |

The T00 distribution tests make core dependency isolation, entry points, and public imports explicit.

## Working-tree boundary

The pre-existing untracked file `pydantic-deepagents-architecture-review.md` was not changed.
