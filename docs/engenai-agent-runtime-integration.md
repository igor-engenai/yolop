# EngenAI Agent Runtime Integration Design

**Date:** 2026-08-17

**Updated:** 2026-08-18

**Status:** Proposed

**Scope:** YoloP package boundaries required by the EngenAI Agent Runtime

## 1. Purpose

This document defines how YoloP supports the EngenAI Agent Runtime without taking ownership of EngenAI application concerns.

The design uses the durable `yolop-runtime` model. It extends that model where production supervision, external-tool continuation, and sparse events need a generic contract. It does not add Absurd or another execution lifecycle around YoloP Runs.

The design adds:

- a generic `RunSupervisor` and the required `Runtime` and `RuntimeStore` contract changes;
- a PostgreSQL implementation of the resulting `RuntimeStore` protocol;
- safe outbound HTTP and OpenAPI capability packages;
- a clean extension boundary for an EngenAI-owned YoloP capability package.

## 2. Decisions

1. The base `yolop` package stays a small stateless kernel around Pydantic AI `AgentSpec`.
2. `yolop-runtime` stays the sole generic owner of durable YoloP Sessions, Runs, pins, messages, events, relations, budgets, claims, leases, and Run supervision.
3. `yolop-postgres-runtime` implements `RuntimeStore` for production PostgreSQL deployments.
4. The runtime Runs table is the durable work queue. No Absurd, Celery, RQ, or second task lifecycle surrounds YoloP Runs.
5. Core can publish an exact immutable AgentSpec. YoloP validates and executes that AgentSpec; it does not pull configuration.
6. `yolop-http` provides safe outbound HTTP mechanics and a read-only Web Fetch capability.
7. `yolop-openapi` provides host-authorized OpenAPI toolsets and depends on `yolop-http`.
8. The embedding host owns OAuth refresh, tenant-isolated token caching, and secret authorization. `yolop-openapi` receives only short-lived prepared authentication values.
9. EngenAI-specific capability code lives in the separate `engenai-yolop` distribution.
10. Authentication, tenant authorization, configuration synchronization, Key Vault access, public APIs, SSE contracts, and outbox/inbox processing stay outside YoloP.

## 3. Package Boundaries

### 3.1 Existing `yolop`

`yolop` continues to own:

- stateless AgentSpec execution;
- capability discovery through `yolop.capabilities`;
- deployment provider allowlists;
- model resolution;
- host-enforced mandatory capabilities;
- `ToolPolicy`;
- immutable trusted tool metadata in `ToolPolicyContext`;
- native Pydantic AI messages, events, results, and deferred-tool contracts.

`ToolPolicyContext` exposes metadata from the code-constructed `ToolDefinition`. AgentSpec and model arguments cannot set or override reserved trusted metadata keys.

It does not gain PostgreSQL, FastAPI, HTTP clients, OpenAPI parsing, EngenAI contracts, or cloud SDKs.

### 3.2 Existing `yolop-runtime`

`yolop-runtime` continues to own the storage-independent durable runtime model:

- namespaced Sessions;
- immutable AgentSpec and model pins;
- idempotent Run reservation;
- Run claims and leases;
- a generic bounded `RunSupervisor`;
- Session serialization;
- canonical and active message histories;
- Run ancestry and root budgets;
- deferred requests, validated results, and continuations;
- durable plugin state;
- live and durable runtime events;
- terminal completion, failure, cancellation, and interruption.

The required generic changes belong in this package:

- store-backed discovery and claiming of accepted Runs;
- lease heartbeat and bounded execution supervision;
- external call results and metadata in `resume_deferred_run(...)`;
- idempotency digests for deferred results;
- host-selected event persistence and bounded live event delivery.

The Runtime accepts host callbacks that resolve the pinned AgentSpec, model, dependencies, and mandatory policy for a claimed Run. It does not know where the host obtained them, and it never calls Core.

### 3.3 New `yolop-postgres-runtime`

`yolop-postgres-runtime` implements the `RuntimeStore` protocol after the generic changes in this design.

It owns:

- PostgreSQL schema and migrations for generic YoloP runtime state;
- atomic Session and Run state transitions;
- idempotent reservation constraints;
- discovery and non-blocking claim of runnable accepted Runs within host-authorized namespaces;
- short `FOR UPDATE SKIP LOCKED` claim transactions or equivalent behavior;
- optional database notifications used only as wake-up hints;
- lease renewal and expiry checks;
- revision and stale-owner fencing checks;
- ordered durable events;
- atomic terminal state and terminal-event commits;
- namespace isolation;
- PgBouncer-compatible Session serialization;
- root-budget accounting;
- bounded plugin state;
- query and index behavior required by the RuntimeStore contract.

No PostgreSQL transaction or checked-out pool connection remains open during model or tool I/O. Session serialization uses durable ownership and fencing, not a long database transaction or a session-level advisory lock.

It does not own:

- EngenAI team, user, Agent, or configuration records;
- JWT validation;
- Core synchronization;
- EngenAI outbox events;
- billing, analytics, or audit projections;
- tool-provider tables;
- public APIs.

The host runs database migrations as an explicit deployment step. Application startup does not run concurrent migrations.

### 3.4 New `yolop-http`

`yolop-http` is an outbound client and capability package. It is not an inbound web server.

It owns:

- an async HTTP client based on an established client library;
- HTTP and HTTPS URL validation;
- host, port, and scheme policy;
- DNS resolution and address validation;
- rejection of loopback, private, link-local, metadata, and other forbidden addresses;
- connection pinning that prevents DNS rebinding while preserving TLS hostname validation;
- scheme, host, port, and DNS revalidation for every redirect hop;
- rejection of any redirect target outside the host-authorized destination set;
- rejection of credentialed cross-origin redirects;
- removal of authorization, API-key, and cookie headers before any permitted non-credentialed origin change;
- bounded redirect count and rejection of unsafe method rewriting;
- explicit host proxy policy, with environment proxy discovery disabled by default;
- connection, read, and total timeouts;
- response-byte and returned-text limits;
- safe content-type handling;
- a read-only `WebFetch` capability for bounded public text retrieval.

It does not expose an unrestricted model tool such as `request(method, url, headers, body)`.

Host policy is authoritative. AgentSpec can select a safe host alias and can narrow allowed destinations, but it cannot broaden host policy.

### 3.5 New `yolop-openapi`

`yolop-openapi` depends on `yolop-http` and follows the host-registry model used by `yolop-mcp`.

It owns:

- a host-owned registry of safe OpenAPI aliases;
- pinned OpenAPI 3 documents;
- host-owned server identities;
- operation discovery;
- mandatory host operation allowlists;
- stricter AgentSpec operation selection;
- bounded parameter and request-body construction;
- JSON, form, JSON-derived, and text request bodies;
- response normalization and limits;
- short-lived prepared authentication values through a host callback;
- model tools equivalent to `<alias>__explore` and `<alias>__call`;
- immutable operation metadata that host `ToolPolicy` can use for denial or approval.

Trusted operation metadata includes the alias, operation ID, HTTP method, normalized path template, and declared effect. The OpenAPI package puts this data in reserved `ToolDefinition.metadata` keys. `ToolPolicyContext` exposes those keys without trusting AgentSpec or model arguments.

The package does not refresh OAuth tokens, cache tokens across calls, or store tenant secrets. The host callback resolves one authorized value for one tenant, credential identity, audience, and scope set. State-changing OpenAPI operations do not follow redirects. Read operations can follow only the bounded same-origin redirect policy from `yolop-http`.

### 3.6 External `engenai-yolop`

`engenai-yolop` is an EngenAI-owned YoloP capability distribution. It is not part of the generic YoloP repository.

It contains capability surfaces such as:

- Form Service tools;
- handoff request contracts and tools;
- external DuckDB pipeline tool definitions;
- bounded knowledge-context formatting;
- outreach query and action tools.

The distribution owns model-visible schemas, descriptions, typed request and result contracts, safe formatting, and Pydantic AI capability or toolset composition.

It uses typed host dependency protocols. It does not import FastAPI, Agent Runtime repositories, Core clients, SQLAlchemy models, Azure SDKs, JWT code, or Key Vault code.

Agent Runtime implements the protocols and owns every real network call, state transition, secret resolution, asynchronous dispatch, and outbox event.

## 4. Dependency Direction

```text
EngenAI Agent Runtime
  ├── yolop-runtime
  │     └── yolop
  ├── yolop-postgres-runtime
  │     └── yolop-runtime
  ├── yolop-context
  ├── yolop-duckdb
  ├── yolop-http
  ├── yolop-openapi
  │     └── yolop-http
  └── engenai-yolop
        └── Pydantic AI capability contracts
```

`yolop` and its generic packages never depend on `engenai-yolop` or Agent Runtime.

## 5. Core-Published AgentSpec

Core may publish the exact AgentSpec that one immutable Agent revision uses.

The AgentSpec contains declarative model data:

- model alias;
- instructions;
- model settings;
- selected capability names;
- safe capability arguments;
- tool descriptions and schemas where the capability contract requires them.

The surrounding EngenAI runtime bundle carries host data that is not AgentSpec configuration:

- team, Agent, and revision identity;
- configuration and AgentSpec digests;
- exact handoff target revisions;
- connection and secret references;
- form binding;
- knowledge manifest;
- mandatory runtime policy and limits;
- revocation identity.

The bundle contains no raw secret, access token, unrestricted Blob credential, or runtime Session identity.

YoloP only validates the supplied AgentSpec against its immutable provider catalog and constructs the Pydantic AI Agent. Agent Runtime resolves host dependencies and mandatory capabilities.

## 6. Runtime Durability

The YoloP Runtime is the durability boundary. A second Agent Runtime queue or message store is not added.

The PostgreSQL store must preserve these properties:

- an accepted Run is committed before execution starts;
- duplicate reservation returns the original Run;
- a claim is short and does not hold a transaction during model or tool I/O;
- one current owner can commit Run state;
- Session order and revision checks prevent concurrent history corruption;
- a client disconnect does not own the Run;
- process-local wake-ups are optimizations only;
- all correctness state survives process termination.

### 6.1 Run supervision

`yolop-runtime` provides one generic `RunSupervisor`. The runtime Runs table is its durable work queue.

The supervisor:

- acquires an execution slot before it claims work;
- claims the oldest eligible accepted Run in a host-authorized namespace set;
- does not claim a Run whose Session predecessor is still active;
- resolves the exact pinned AgentSpec, model, dependencies, and mandatory policy through host callbacks;
- starts a lease heartbeat before model or tool I/O;
- renews the lease at a bounded fraction of its duration;
- stops execution when renewal fails or ownership is fenced;
- reports resolver and execution failures with stable error codes;
- stops new claims during shutdown, drains for a bounded period, and interrupts remaining owned Runs.

A process-local notification can wake the supervisor after reservation. The supervisor also polls durable accepted Runs, so a crash between reservation and notification cannot strand work. PostgreSQL notifications can reduce polling delay, but they are never required for correctness.

Only the supervisor executes accepted cloud Runs. Request and SSE handlers reserve or observe Runs; they do not own execution tasks. The host remains responsible for admission limits and for the authorized namespace set supplied to the supervisor.

### 6.2 Current interruption behavior

The current SQLite store changes an expired or shutdown-owned running Run to `INTERRUPTED`. This is durable evidence of an incomplete Run, but it is not automatic execution recovery.

An accepted Run that never started can be claimed by another supervisor. A running Run whose owner disappears becomes `INTERRUPTED` after lease expiry. Automatic retry of that interrupted Run requires an explicit, bounded YoloP Runtime recovery policy. It must not be implemented as a separate Agent Runtime queue.

A recovery policy must define:

- which interruption classes are retryable;
- attempt and wall-clock limits;
- the committed message and tool checkpoint used for restart;
- provider-call replay behavior;
- stable side-effect idempotency identities;
- stale-owner fencing;
- the terminal state after exhaustion.

Durable pickup of accepted Runs and lease renewal are unconditional production launch gates. Retry of interrupted Runs is a launch gate only when the host promises completion after process loss.

### 6.3 External tools

Pydantic AI `ExternalToolset` and `DeferredToolResults` are the generic asynchronous-tool boundary.

YoloP stores every deferred request with its originating Run and tool-call identity before the host dispatches work. The host records invocation state under the stable identity `(namespace, originating_run_id, tool_call_id)` and uses that identity as the external side-effect idempotency key where the provider supports one.

`Runtime.resume_deferred_run(...)` accepts Pydantic AI call results, approvals, and bounded metadata. It validates that every supplied tool-call ID belongs to the pending request, rejects missing or unknown required results, and validates each result against the host-owned typed contract before model execution.

The continuation reservation includes a canonical digest of the validated calls, approvals, and metadata. Reuse of an idempotency key with the same digest returns the original continuation. Reuse with a different digest raises `idempotency_conflict`.

The host owns dispatch and its invocation table. YoloP owns deferred request persistence and continuation. YoloP does not add a generic external-job package.

## 7. Event and Message Storage

A production PostgreSQL adapter must not copy the SQLite physical design without measurement.

### 7.1 Events

The current Runtime persists every native stream event. This gives exact replay but can create excessive writes for token deltas.

The Runtime supports a host-selected persistence policy with two paths:

- selected semantic events are appended to durable storage;
- every event can be offered to a bounded live event sink.

The Runtime emits lifecycle events for reservation, claim, start, terminal completion, failure, cancellation, and interruption. It also exposes native agent and extension events. An EngenAI host durably selects lifecycle, tool, deferred, terminal, and final-output events. The Runtime appends a selected event before it offers that event to the live sink. Terminal state and its terminal event commit in one store transaction. A durable append failure aborts the state transition and fails execution; the Runtime never hides loss of required state.

The live sink is a bounded, non-blocking broker interface. It is not an SSE socket writer. A sink offer has a short fixed bound and cannot wait on client network I/O. Overflow or sink detachment can drop transient token and thinking deltas. It cannot cancel or fail the Run. The host records drop counters and sink health so this degradation is observable.

The host drains the broker into the current SSE connection. Disconnecting the client detaches only that drain. Durable semantic events remain available for replay.

Only durable events receive durable sequence numbers. Transient live events use connection-local ordering and never become reconnect cursors. Reconnection first reads the current Run snapshot and final output, then replays durable events after the last durable cursor. It does not promise old token boundaries.

### 7.2 Messages

The PostgreSQL schema must avoid a full transcript copy for every Run.

The preferred physical model is:

- canonical append-only Session messages;
- Run references to input and output message ranges;
- one bounded active-context or compaction snapshot per Session revision;
- immutable terminal Run output and usage;
- no repeated full-history JSON payload in each Run row.

The adapter can reconstruct RuntimeStore snapshots at the protocol boundary. If that reconstruction is too expensive or the protocol forces excessive duplication, change the RuntimeStore contract before production promotion.

## 8. Security Boundary

YoloP packages own reusable enforcement mechanics only.

The embedding host owns:

- user and service authentication;
- business authorization;
- tenant namespace selection;
- configuration publication and synchronization;
- provider and capability deployment allowlists;
- Key Vault and managed identity;
- secret-reference authorization;
- OAuth refresh and tenant-isolated token caching;
- token cache keys that include namespace, credential identity, audience, and scopes, never only an OpenAPI alias;
- tool invocation signatures;
- public request and SSE policy;
- business audit and outbox events.

A client cannot select a Runtime namespace, resource path, server URL, secret reference, or installed provider directly.

## 9. EngenAI Host Flow

```text
Core publishes immutable runtime bundle and AgentSpec
  → Core outbox event
  → Agent Runtime pulls and validates the bundle
  → Agent Runtime stores its local read model and inbox cursor
  → request token selects an exact local bundle
  → request handler authorizes a namespace and reserves a Run
  → process-local wake-up hints the yolop-runtime RunSupervisor
  → RunSupervisor claims durable accepted work from yolop-postgres-runtime
  → Agent Runtime callbacks resolve the exact pinned bundle and dependencies
  → RunSupervisor executes with lease heartbeat and stale-owner fencing
  → Runtime offers live events to a bounded host broker
  → current SSE connection drains the broker without owning the Run
  → durable state supports snapshots and reconnect
  → Agent Runtime projects business events into its own outbox
```

YoloP does not call Core and does not publish EngenAI outbox events.

## 10. Explicit Exclusions

This design does not add these generic YoloP packages:

- `yolop-handoffs`;
- `yolop-knowledge`;
- `yolop-duckdb-pipeline`;
- `yolop-external-tools`;
- `yolop-absurd`.

Handoff execution, knowledge retrieval, stateful Blob pipelines, and outreach state remain EngenAI concerns exposed through `engenai-yolop` capability contracts.

Direct generic PostgreSQL and MySQL query capabilities are also excluded. Applications should expose constrained domain tools. The existing `yolop-duckdb` package remains the generic read-only SQL capability.

## 11. Validation Gates

The packages are ready for an EngenAI production cutover only when tests prove:

1. PostgreSQL RuntimeStore contract parity with SQLite for the extended contract.
2. Atomic idempotent reservation under concurrent requests.
3. A process loss after reservation but before wake-up does not strand an accepted Run.
4. The supervisor does not claim beyond capacity or run two active predecessors for one Session.
5. Lease heartbeat keeps a long model or tool call owned, and stale owners cannot commit.
6. Claim, revision, Session serialization, and fencing work under concurrent processes and PgBouncer.
7. Session and claim discovery stay within host-authorized namespaces.
8. Supervisor shutdown drains or interrupts every owned Run with a stable terminal state.
9. Client disconnect, live-sink failure, and live-sink overflow do not cancel execution.
10. Live event overload is bounded and observable; it cannot create an unbounded memory queue.
11. Deferred external calls, approvals, metadata, and tool-call IDs are validated before continuation.
12. Reusing a continuation idempotency key with a different external result is rejected.
13. Deferred external-tool continuation survives restart without duplicate side effects.
14. Sparse durable events recover current state and final output after lost live deltas.
15. Message storage remains bounded under long Sessions and compaction.
16. HTTP and OpenAPI transports reject SSRF and DNS-rebinding attempts on every redirect hop.
17. Credentialed cross-origin redirects cannot disclose authorization, API-key, or cookie values.
18. AgentSpec cannot broaden host connection, tool, metadata, or secret policy.
19. `ToolPolicy` receives immutable operation identity and effect metadata for OpenAPI calls.
20. OAuth tokens cannot cross namespace, credential, audience, or scope boundaries.
21. Production-like PostgreSQL, PgBouncer, supervisor, event-broker, and SSE load meets the host targets.

## 12. Non-Goals

This design does not define:

- EngenAI public API compatibility;
- migration of existing Agent Runtime Session history;
- Core domain models or UI;
- Azure deployment topology;
- workflow DAG execution;
- arbitrary tenant code or plugins;
- raw database access for language models.

Those concerns belong to the EngenAI architecture and cutover plan.
