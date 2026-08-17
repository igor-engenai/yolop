# EngenAI Agent Runtime Integration Design

**Date:** 2026-08-17

**Status:** Proposed

**Scope:** YoloP package boundaries required by the EngenAI Agent Runtime

## 1. Purpose

This document defines how YoloP supports the EngenAI Agent Runtime without taking ownership of EngenAI application concerns.

The design uses the existing durable `yolop-runtime` facade. It does not add Absurd or another execution lifecycle around YoloP runs.

The design adds:

- a PostgreSQL implementation of the existing `RuntimeStore` protocol;
- safe outbound HTTP and OpenAPI capability packages;
- a clean extension boundary for an EngenAI-owned YoloP capability package.

## 2. Decisions

1. The base `yolop` package stays a small stateless kernel around Pydantic AI `AgentSpec`.
2. `yolop-runtime` stays the sole generic owner of durable YoloP Sessions, Runs, pins, messages, events, relations, budgets, claims, and leases.
3. `yolop-postgres-runtime` implements `RuntimeStore` for production PostgreSQL deployments.
4. No Absurd, Celery, RQ, or second task lifecycle surrounds YoloP Runs.
5. Core can publish an exact immutable AgentSpec. YoloP validates and executes that AgentSpec; it does not pull configuration.
6. `yolop-http` provides safe outbound HTTP mechanics and a read-only Web Fetch capability.
7. `yolop-openapi` provides host-authorized OpenAPI toolsets and depends on `yolop-http`.
8. EngenAI-specific capability code lives in the separate `engenai-yolop` distribution.
9. Authentication, tenant authorization, configuration synchronization, Key Vault access, public APIs, SSE contracts, and outbox/inbox processing stay outside YoloP.

## 3. Package Boundaries

### 3.1 Existing `yolop`

`yolop` continues to own:

- stateless AgentSpec execution;
- capability discovery through `yolop.capabilities`;
- deployment provider allowlists;
- model resolution;
- host-enforced mandatory capabilities;
- `ToolPolicy`;
- native Pydantic AI messages, events, results, and deferred-tool contracts.

It does not gain PostgreSQL, FastAPI, HTTP clients, OpenAPI parsing, EngenAI contracts, or cloud SDKs.

### 3.2 Existing `yolop-runtime`

`yolop-runtime` continues to own the storage-independent durable runtime model:

- namespaced Sessions;
- immutable AgentSpec and model pins;
- idempotent Run reservation;
- Run claims and leases;
- Session serialization;
- canonical and active message histories;
- Run ancestry and root budgets;
- deferred continuations;
- durable plugin state;
- runtime events;
- terminal completion, failure, cancellation, and interruption.

The Runtime accepts the AgentSpec and host dependencies for each execution. It does not know where the host obtained them.

### 3.3 New `yolop-postgres-runtime`

`yolop-postgres-runtime` implements the existing `RuntimeStore` protocol.

It owns:

- PostgreSQL schema and migrations for generic YoloP runtime state;
- atomic Session and Run state transitions;
- idempotent reservation constraints;
- non-blocking claims;
- lease renewal and expiry checks;
- revision and fencing checks;
- ordered durable events;
- namespace isolation;
- Session locks;
- root-budget accounting;
- bounded plugin state;
- query and index behavior required by the RuntimeStore contract.

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
- safe redirect handling;
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
- secret-reference resolution through a host callback;
- bounded OAuth token refresh and caching;
- model tools equivalent to `<alias>__explore` and `<alias>__call`;
- operation metadata that host `ToolPolicy` can use for denial or approval.

The package never stores tenant secrets. It receives short-lived resolved values from the host only while a connection is used.

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

The existing YoloP Runtime is the durability boundary. A second Agent Runtime queue or message store is not added.

The PostgreSQL store must preserve these properties:

- an accepted Run is committed before execution starts;
- duplicate reservation returns the original Run;
- a claim is short and does not hold a transaction during model or tool I/O;
- one current owner can commit Run state;
- Session order and revision checks prevent concurrent history corruption;
- a client disconnect does not own the Run;
- process-local wake-ups are optimizations only;
- all correctness state survives process termination.

### 6.1 Current interruption behavior

The current SQLite store changes an expired or shutdown-owned running Run to `INTERRUPTED`. This is durable evidence of an incomplete Run, but it is not automatic execution recovery.

The PostgreSQL implementation must first match the public RuntimeStore contract. Automatic retry after process loss must be added only through an explicit, bounded YoloP Runtime recovery policy. It must not be implemented as a separate Agent Runtime queue.

A recovery policy must define:

- which interruption classes are retryable;
- attempt and wall-clock limits;
- the committed message and tool checkpoint used for restart;
- provider-call replay behavior;
- stable side-effect idempotency identities;
- stale-owner fencing;
- the terminal state after exhaustion.

This policy is a production launch gate when the host promises completion after process loss.

### 6.2 External tools

Pydantic AI `ExternalToolset` and `DeferredToolResults` are the generic asynchronous-tool boundary.

YoloP stores the deferred request and continuation history. The host dispatches the external operation, records its own invocation state, and resumes the YoloP continuation with a validated result.

YoloP does not add a generic external-job package.

## 7. Event and Message Storage

A production PostgreSQL adapter must not copy the SQLite physical design without measurement.

### 7.1 Events

The current Runtime persists every native stream event. This gives exact replay but can create excessive writes for token deltas.

The Runtime must support a host-selected persistence policy with two paths:

- every native event can go to a live event sink;
- only selected semantic events need durable storage.

An EngenAI host persists lifecycle, tool, deferred, terminal, and final-output events. Token deltas are best-effort live data. Reconnection always recovers current state and final output, but it does not promise exact old token boundaries.

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
  → Agent Runtime resolves host dependencies and mandatory capabilities
  → yolop-runtime reserves and executes against yolop-postgres-runtime
  → native live events feed the current SSE connection
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

1. PostgreSQL RuntimeStore contract parity with SQLite.
2. Atomic idempotent reservation under concurrent requests.
3. Claim, lease, revision, and stale-owner fencing under concurrent processes.
4. Session isolation across namespaces.
5. Process termination preserves accepted and running state according to the selected recovery policy.
6. Client disconnect does not cancel execution.
7. Deferred external-tool continuation survives restart without duplicate side effects.
8. Sparse durable events recover current state and final output after lost live deltas.
9. Message storage remains bounded under long Sessions and compaction.
10. HTTP and OpenAPI transports reject SSRF and DNS-rebinding attempts.
11. AgentSpec cannot broaden host connection, tool, or secret policy.
12. Production-like PostgreSQL, PgBouncer, and SSE load meets the host targets.

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
