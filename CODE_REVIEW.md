# Code Review: RAG Evaluator Platform

**Date:** 2026-06-19
**Reviewer:** Claude (automated code review)
**Scope:** Full repository — Core CLI (`src/rag_evaluator`), Backend API (`platform/backend`), Frontend (`platform/frontend`)
**Commit reviewed:** `c6f8462` (branch `claude/codebase-code-review-act7x3`)

---

## 1. Executive Summary

RAG Evaluator is a sizable, multi-layered platform (~42k LOC) for designing, indexing, and evaluating
RAG implementations. It is composed of three cooperating layers:

| Layer | Location | Approx. LOC | Tech |
|-------|----------|-------------|------|
| Core / CLI | `src/rag_evaluator` | ~15.9k | Python, LangChain, DeepEval, Chroma/Qdrant/Neo4j |
| Backend API | `platform/backend/app` | ~14.5k | FastAPI, SQLModel/SQLAlchemy, async, Alembic |
| Frontend | `platform/frontend/src` | ~12.0k | React, Vite, TypeScript, Tailwind |

**Overall assessment:** This is a well-structured, ambitious codebase with a clear separation of concerns,
a clean RAG abstraction (`BaseRAG`), consistent use of typed configuration (Pydantic), structured logging,
proper async patterns, and a healthy backend test suite (242 backend test functions, 113 core test functions).
The engineering maturity is above average for an alpha-stage project.

However, there are **several material security concerns** that should be addressed before any
non-local/multi-tenant deployment, the most important being **(a) arbitrary code execution of
LLM-generated Python in-process by default** in the RLM RAG, **(b) complete absence of authentication/
authorization** on the backend API, and **(c) SSRF exposure** in the user-configurable webhook delivery
path. None of these are blockers for the project's current "local evaluation tool" framing, but they are
landmines if the Platform is ever exposed beyond `localhost`.

**Top priorities:**
1. Sandbox or gate the RLM code-execution paths (Critical).
2. Add authentication/authorization to the backend, or document loudly that it is single-user/localhost-only (High).
3. Add SSRF protections to webhook delivery (High).
4. Expand CI to cover the backend and frontend, not just core (Medium).

---

## 2. Architecture Overview

The design pattern is sound and consistent:

- `BaseRAG` (`src/rag_evaluator/common/base_rag.py`) defines a clean abstract interface with both a
  legacy `query()` and a more granular `retrieve()` + `generate()` split. This split is genuinely useful —
  it enables caching retrieval, re-running generation experiments, and standardized tracing.
- Implementations are pluggable via a registry (`rag_implementations/registry.py`): `vector_semantic`
  (Chroma), `vector_hybrid` (Qdrant + SPLADE + RRF), `graph_rag` (Neo4j), `filesystem_rag` (agentic),
  and `rlm_rag` (REPL/agentic).
- The backend wraps the core via an adapter layer (`services/rag_adapter.py`), cleanly separating
  build-time config from query-time overrides (`EffectiveRAGConfig`). This is a thoughtful design that
  correctly keeps API keys out of stored config snapshots (`_apply_provider_credentials`,
  `rag_adapter.py:111`).
- Backend follows a conventional FastAPI layout: `api/` (routers) → `services/` (business logic) →
  `models/` (SQLModel) + `schemas/` (Pydantic DTOs). Separation is clean and consistent.

This layering is the codebase's biggest strength: the core is usable standalone, and the platform
composes it rather than duplicating it.

---

## 3. Strengths

- **Clean abstraction & extensibility.** Adding a new RAG type is a well-defined operation thanks to
  `BaseRAG` + the registry. The `retrieve()`/`generate()` separation is forward-looking.
- **Typed configuration everywhere.** Both `src/rag_evaluator/config.py` and
  `platform/backend/app/config.py` use `pydantic-settings` with sensible defaults and `extra="ignore"`.
  Secrets default to empty rather than hardcoded.
- **Structured logging.** The backend uses a structured logger with request-ID propagation middleware
  (`main.py:107`) and a global exception handler that avoids leaking internals to clients (`main.py:171`).
- **Solid error contract.** Consistent `ErrorResponse` schema, `AppException` handling, and request-ID
  echoing make the API debuggable.
- **Resumable/checkpointed indexing.** `CheckpointStore` and `prepare_documents_resumable` show real
  attention to long-running-job robustness, including reconciling interrupted builds on startup
  (`main.py:71`).
- **Thread-safe token accounting.** `BaseRAG` uses thread-local token tracking with aggregation to a
  global total, with a dedicated thread-safety test (`tests/unit/test_token_usage_thread_safety.py`).
- **Good test breadth on the backend** — API, services, lifecycle, error handling, and checkpointing are
  all covered.
- **Defense-in-depth intent in the agentic RAGs.** The filesystem tools implement path-escape checks,
  and the RLM has an explicit injection-guard module — the *intent* is right (caveats in §4).

---

## 4. Security Findings

### 4.1 [Critical] LLM-generated code is `exec()`'d in-process by default (RLM RAG)

`RLMConfig.security_mode` defaults to `"lite"` (`rlm_rag/rlm_rag.py:38`), and in lite mode the agent's
REPL compiles and executes model-produced Python **directly in the host process** with no sandbox:

```python
# src/rag_evaluator/rag_implementations/rlm_rag/agent.py:630
compiled = compile(code, "<repl>", "exec")
exec(compiled, self.namespace)          # arbitrary code, in-process
...
result = eval(last_line, self.namespace)  # agent.py:589
```

Because the LLM chooses what code to run and documents are attacker-influenceable (prompt injection),
this is a path to arbitrary code execution on the host. The `InjectionGuard` regex list
(`security.py:42`) is a best-effort heuristic, not a security boundary, and is **off by default**
(`enable_detection=False`) and only wired up in `"full"` mode anyway.

Even the `"full"` mode `ProcessREPL` is **not a real sandbox**: it runs in a subprocess with a timeout
and a curated namespace, but `exec(compiled, namespace)` (`security.py:316`) leaves Python's real
`__builtins__` reachable, so `__import__('os').system(...)` still works — the curated builtins dict does
not remove access. The subprocess gives you a kill-switch and memory isolation, not capability isolation.

**Recommendation:** Treat this as untrusted-code execution. At minimum:
- Make the secure mode the default, and document the risk prominently.
- In the subprocess, strip builtins (`exec(code, {"__builtins__": safe_builtins})`) and run under OS-level
  sandboxing (seccomp/landlock/nsjail, a container with no network + read-only FS, or a gVisor/Firecracker
  microVM). A timeout alone is insufficient.
- Consider an allowlist AST validator that rejects imports, attribute access to dunders, etc., before
  execution.

### 4.2 [High] No authentication or authorization on the backend API

There is no auth dependency anywhere in `app/api/deps.py` (only pagination/filter helpers) and no auth
middleware in `main.py`. Every endpoint — create/delete projects, run evaluations, register webhooks,
trigger index builds (all of which spend money on LLM calls and can read the filesystem) — is fully open.
CORS is configured with `allow_credentials=True` and `allow_origins` defaulting to `http://localhost:3000`,
which is fine locally but underscores that the app assumes a trusted single user.

**Recommendation:** Add at least an API-key/bearer dependency on mutating routes, or — if single-user
localhost is the intended deployment model — state that explicitly in the README/deployment docs and bind
to `127.0.0.1`. Project-scoped resources also currently have no tenant isolation.

### 4.3 [High] SSRF via webhook delivery

`WebhookService._deliver_webhook` POSTs to a fully user-controlled `webhook.url`
(`services/webhook_service.py:114`) with no validation that the destination is external/public. Combined
with no auth (§4.2), an actor can register a webhook pointing at `http://169.254.169.254/...` or internal
services and use the platform as an SSRF pivot. The `send_test_event` path has the same issue.

**Recommendation:** Validate webhook URLs (scheme allowlist `https`/`http`, resolve and block
RFC1918/loopback/link-local/metadata ranges, optionally an allowlist), disable redirects on the client,
and keep the existing HMAC signing (which is good).

### 4.4 [Low/Medium] Path-traversal defense in filesystem tools — looks correct, keep it tested

`FilesystemRAGTools._resolve_path` (`filesystem_rag/agent/tools.py:37`) resolves and then checks
`relative_to(prepared_path.resolve())`, which is the right pattern and should defeat `../` escapes and
absolute paths. This is good. Two notes: (1) `grep_search`/`find_files` use `rglob` from the resolved
root, which is safe, but they do not re-check followed **symlinks** that point outside the root — worth a
test; (2) ensure the `prepared_path` itself is never attacker-controlled.

### 4.5 [Low] Injection-pattern heuristics may give false confidence

The `INJECTION_PATTERNS` list (`security.py:42`) and `risk_score = matched/3` heuristic
(`security.py:151`) will both miss real attacks and flag benign text. It's fine as telemetry/logging, but
should not be presented to users as a security control. Document it as best-effort.

---

## 5. Backend Findings (`platform/backend`)

- **[Cleanup] Stray empty file committed.** `platform/backend/file` is a 0-byte file that looks
  accidental (likely a shell redirect artifact). Remove it.
- **[Medium] Broad exception handling is pervasive.** ~98 `except Exception` sites across `src` +
  `app`. Many are legitimate (background jobs, webhook delivery), but blanket catches can mask bugs and
  swallow `asyncio.CancelledError` semantics. Audit hot paths (evaluation runner, index builder) to catch
  narrower types and always log with stack context (the structured `logger.exception` usage in `main.py`
  is the right model).
- **[Medium] Sync-over-async bridge in DeepEval wrapper.** `SafeDeepEvalLLM.generate` spins up a fresh
  event loop per call (`evaluation_runner.py:65`). This works because it runs inside a thread executor,
  but creating/closing a loop per metric call is costly and fragile if ever called from the main loop.
  Consider a single dedicated loop/thread or an explicit `asyncio.run_coroutine_threadsafe` against a
  known loop.
- **[Low] Webhook event filtering in Python.** `trigger_event` loads all active webhooks then filters by
  event in Python (`webhook_service.py:66`) because events are a JSON list. Fine at small scale; if
  webhook volume grows, push the filter into the query (JSON containment for Postgres).
- **[Low] Singletons via module globals.** `get_webhook_service()` / `get_rag_adapter_service()` use
  mutable module-level singletons. Acceptable for a single-process app, but they complicate testing and
  multi-worker deployments (each worker gets its own cache/client). Lifespan already disposes them, which
  is good.
- **[Positive] Startup reconciliation** of interrupted index builds (`main.py:71`) is a nice robustness
  touch.

---

## 6. Core / CLI Findings (`src/rag_evaluator`)

- **[Medium] `BaseRAG._token_usage` monkeypatches instance methods.** The thread-local wrapper rebinds
  `add_prompt_tokens`/etc. on each thread's `TokenUsage` instance (`base_rag.py:127-145`). The code itself
  flags this as "a bit hacky." It works and is tested, but it's surprising and brittle (e.g., anything
  that introspects or re-wraps those methods will compound). A cleaner design: have `TokenUsage` take an
  optional parent/aggregator in its constructor and forward there, instead of rebinding methods at
  runtime.
- **[Low] Default `retrieve()` fabricates scores.** The backward-compat default assigns
  `score = 1.0 - i*0.1` by rank (`base_rag.py:249`). This produces misleading trace data (and can go
  negative beyond rank 10) for any implementation that doesn't override `retrieve()`. Consider `None`/
  "unknown" scores rather than synthetic ones, so downstream charts don't treat them as real.
- **[Low] Function-local imports.** Several methods import inside the body (`base_rag.py:223`,
  `webhook_service.py:143`). Usually done to avoid cycles or heavy import cost; if it's cycle-avoidance,
  a `TYPE_CHECKING` guard or module restructure is cleaner. Minor.
- **[Low] `print()` vs logging.** ~176 `print()` calls in `src`. The bulk are legitimately in the CLI and
  Streamlit UI (user-facing output), but spot-check that none live in library/retrieval code paths where
  structured logging is expected.
- **[Positive] The RLM/filesystem agents are well-documented** with module docstrings and clear tool
  schemas (`tools.py:403` OpenAI function defs), which materially helps maintainability.

---

## 7. Frontend Findings (`platform/frontend`)

- **[Low] No auth token handling in the API client.** `api/client.ts` sets only `Content-Type` and a
  request ID (`client.ts:5-16`). Consistent with the backend having no auth (§4.2); revisit together.
- **[Low] Error handling logs to console only.** The response interceptor (`client.ts:19`) logs and
  rejects, which is reasonable, but ensure user-facing surfaces consistently render the toast/error
  rather than failing silently.
- **[Medium] No frontend tests.** Zero `*.test.ts(x)` files were found. For ~12k LOC of TypeScript with
  non-trivial logic (comparison/diff utilities, streaming hooks like `useEvaluationStream`), some unit
  coverage of the pure utilities (`compare-utils.ts`, `lib/utils.ts`) would catch regressions cheaply.
- **[Positive] Component organization is clean** — feature-foldered (`comparisons/`, `evaluations/`,
  `test-sets/`, …) with a shared `ui/` primitives layer and a typed API client.

---

## 8. Testing & CI

- **Test breadth is good** on the Python side: 242 backend + 113 core test functions, plus an
  `integration/` suite for each RAG type. Lifecycle, checkpointing, error handling, and quality gates are
  all covered.
- **[Medium] CI only covers the core package.** `.github/workflows/tests.yml` runs `uv run pytest`,
  `ruff check .`, and `mypy src/rag_evaluator` — all from the repo root. It does **not**:
  - run the **backend** test suite (`platform/backend/tests`, the larger suite),
  - run **frontend** lint/typecheck/build (`npm run lint`, `tsc`),
  - type-check the backend (`mypy` is scoped to `src/rag_evaluator` only).

  So the majority of the codebase's tests never run in CI. Add backend and frontend jobs (matrix or
  separate jobs), each `cd`-ing into the right directory per the project's own AI-collaboration rules.
- **[Low] CI triggers only on `main`.** Pushes/PRs to feature branches that don't target `main` won't run
  CI. Consider broadening `pull_request` triggers.
- **[Low] No coverage gate.** `pytest-cov` is a dependency but CI doesn't enforce a threshold or publish
  coverage.

---

## 9. Documentation & Configuration

- **[Positive] Strong docs surface.** `README.md`, `CLAUDE.md`/`AGENTS.md`/`GEMINI.md`, `CONTRIBUTING.md`,
  and a `docs/` map (cli/api/deployment) are present and coherent. `.env.example` documents configuration.
- **[Low] Three near-duplicate agent-instruction files** (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`) risk
  drift. Consider a single canonical file referenced by the others.
- **[Low] Two `Settings` classes with divergent conventions.** Core uses lowercase fields
  (`openai_api_key`) and resolves `.env` from project root; backend uses uppercase (`OPENAI_API_KEY`) and
  a relative `["./.env", "../../.env"]` search. Both are reasonable, but the casing/loading divergence is
  a footgun when sharing one root `.env`. Document the precedence clearly.
- **[Low] Planning/scratch docs in repo root.** `PLAN_RAG_PLAYGROUND.md` and similar planning artifacts
  live at the top level; consider moving to `docs/plans/` to keep the root clean.
- **[Note] CLAUDE.md says Windows/PowerShell** is the dev environment; the CI runs Ubuntu. Ensure
  cross-platform path handling (the filesystem RAG uses `pathlib`, which is good) is exercised on both.

---

## 10. Prioritized Recommendations

| # | Priority | Area | Recommendation |
|---|----------|------|----------------|
| 1 | **Critical** | RLM RAG | Treat REPL execution as untrusted code: strip `__builtins__`, add OS-level sandboxing, make secure mode the default, document the risk. (§4.1) |
| 2 | **High** | Backend | Add authentication/authorization (or bind to localhost + document single-user model) and add tenant isolation for project-scoped data. (§4.2) |
| 3 | **High** | Webhooks | Add SSRF protections: block internal/metadata IP ranges, allowlist schemes, disable redirects. (§4.3) |
| 4 | **Medium** | CI | Extend CI to run backend tests, frontend lint/build, and backend type-checking. (§8) |
| 5 | **Medium** | Core | Replace the monkeypatch-based token aggregation with a constructor-injected aggregator. (§6) |
| 6 | **Medium** | Backend | Audit broad `except Exception` handlers in hot paths; narrow and ensure stack logging. (§5) |
| 7 | **Low** | Cleanup | Remove the stray `platform/backend/file`; consolidate duplicate agent docs. (§5, §9) |
| 8 | **Low** | Core | Stop fabricating synthetic retrieval scores in the default `retrieve()`. (§6) |
| 9 | **Low** | Frontend | Add unit tests for pure utilities (`compare-utils.ts`, `lib/utils.ts`). (§7) |
| 10 | **Low** | Security | Reframe injection-pattern detection as telemetry, not a control; add a symlink-escape test for filesystem tools. (§4.4, §4.5) |

---

### Closing note

The bones of this project are good: a clean RAG abstraction, sensible layering, typed config, structured
logging, and real test discipline on the backend. The headline risks all stem from the same implicit
assumption — that this runs as a **trusted, single-user, local tool**. That assumption is fine as long as
it's made explicit and enforced (localhost binding, secure-by-default code execution). The moment the
Platform is exposed to other users or a network, items 1–3 become urgent. Addressing those, plus closing
the CI gap, would move this from a strong alpha to something deployable with confidence.
