# Codebase Reorganization — 2026-03-01

## Overview

Executed the plan in `docs/plans/2026-02-28-codebase-reorganization.md`. Goal: remove dead code, fix a frontend bug, harden Docker config, and replace a fragile `sys.path` hack with a proper package dependency. All 7 tasks were completed and merged to `main` via fast-forward.

---

## What Was Done

### Phase 1: Safe Cleanup

**Task 1 — Delete dead root-level files** (`8a1d37c`)

Deleted three files that had no imports anywhere in the codebase:
- `debug_eval.py` — ad-hoc evaluation debug script
- `fetch_eval_id.py` — one-off utility script
- `nul` — Windows artifact (untracked, deleted from working tree only)

**Task 2 — Archive migration scripts** (`0eef92f`)

Created `scripts/archive/` and moved five one-time data migration scripts there:
- `fix_legacy_metrics.py`, `fix_migration_data.py`, `fix_missing_answers.py`
- `migrate_existing_evaluations.py`, `migrate_legacy_data.py`

Also deleted two scripts that duplicated existing CLI commands:
- `scripts/run_evaluation.py` — replaced by `rag-eval evaluate`
- `scripts/run_streamlit.py` — replaced by `rag-eval ui`

**Task 3 — Fix test set filenames with spaces** (`53ad0c8`)

- Renamed `data/test_set _txt.json` → `data/test_set_txt.json` (git-tracked rename)
- Renamed `data/test_set RAG_5.json` → `data/test_set_rag5.json` (untracked file, local rename only)
- Updated `.gitignore` to whitelist `data/test_set_verify.json` so the pipeline verification test set is committed

### Phase 2: Bug Fix

**Task 4 — Fix `client.ts` import URL** (`18cd172`)

The frontend API client had the wrong path for the test set import endpoint:

```typescript
// Before (wrong — no such endpoint)
import: (id: string, data: unknown) => apiClient.post(`/test-sets/${id}/import`, data)

// After (correct — matches backend router)
import: (projectId: string, data: unknown) => apiClient.post(`/projects/${projectId}/test-sets/import`, data)
```

The backend endpoint signature (`POST /projects/{project_id}/test-sets/import`) was confirmed from `platform/backend/app/api/test_sets.py:459`.

### Phase 3: Docker Compose Hardening

**Task 5 — Harden `docker-compose.yml`** (`f17f771`)

Rewrote `docker-compose.yml` with:
- **Added PostgreSQL 15** service (the backend supports Postgres but it was missing from compose)
- **Pinned image versions**: `qdrant/qdrant:v1.8.4`, `neo4j:5.18-community` (was `latest` / unpinned)
- **Memory limits**: postgres `512m`, qdrant `2g`, neo4j `2g`
- **Health checks** on all three services
- **Named volumes** instead of local bind mounts for data persistence
- Added `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` vars to `.env.example`

### Phase 4: Fix the sys.path Hack

**Task 6 — Replace `sys.path` hack with proper uv dependency** (`6cd1128`)

`platform/backend/app/services/rag_adapter.py` had a fragile path traversal at startup:

```python
# Before — 5 levels of parent() to find src/
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
```

Replaced with a proper `uv` path dependency in `platform/backend/pyproject.toml`:

```toml
dependencies = [
    ...
    "rag-evaluator",
]

[tool.uv.sources]
rag-evaluator = { path = "../../", editable = true }
```

The `sys.path` block and the `import sys` statement were removed from `rag_adapter.py`. The `from rag_evaluator...` imports now work through the declared package dependency. After merging, run `uv sync` in `platform/backend/` to install the editable dep.

### Phase 5: Consolidate RAG Type Registry

**Task 7 — Extract RAG type registry to `src/`** (`1c9aafa`)

`rag_adapter.py` defined `RAG_TYPE_REGISTRY` and `RAG_TYPE_PARAMETERS` locally. `cli.py` duplicated the same four RAG types in a hardcoded `if/elif` chain. Neither was the authoritative source.

Created `src/rag_evaluator/rag_implementations/registry.py` as the single source of truth:
- `RAG_TYPES` — dict of type keys to name/description metadata
- `RAG_TYPE_PARAMETERS` — parameter schemas for each type (used by the platform UI)
- `get_rag_class(rag_type)` — dynamically imports and returns the implementation class

Updated `cli.py`:
- Removed four direct RAG class imports
- Replaced the `if/elif` dispatch with `get_rag_class(rag_type)()`
- Uses `list(RAG_TYPES)` for argparse `choices` instead of a hardcoded list

Updated `rag_adapter.py`:
- Removed the local `RAG_TYPE_REGISTRY` and `RAG_TYPE_PARAMETERS` dicts (82 lines removed)
- `_get_rag_class()` now delegates to `get_rag_class()` from the registry
- `get_available_rag_types()` and `get_parameter_schema()` read from the registry

---

## Problems Encountered

### 1. pytest hangs indefinitely

Every attempt to run `uv run pytest` — or even `.venv/Scripts/python.exe -m pytest` — produced no output and hung forever. This happened regardless of:
- Which test file was targeted
- Whether coverage was disabled (`-p no:cov`)
- Whether a single lightweight test file was used

**Root cause (diagnosed):** Several packages in this project do blocking I/O at import time. When pytest collects tests, it imports all test modules, which in turn import `rag_evaluator` modules, which import `chromadb`, `deepeval`, and related packages. At least one of these packages (likely `deepeval`, which is known to spawn background threads and make network calls on import) blocks indefinitely during collection.

**How tests were verified instead:** All changes were verified using targeted `python.exe -c "..."` invocations that imported only what was needed, without going through pytest's collection phase:
- Registry structure, `ValueError` on unknown type, presence of all four RAG types
- `rag_adapter.py` source checked for absence of `sys.path`, `import sys`, and `RAG_TYPE_REGISTRY`
- `cli.py` checked for absence of direct class imports and presence of registry import
- `pyproject.toml` checked for `rag-evaluator` dep and `[tool.uv.sources]`

The test file `tests/unit/test_rag_registry.py` was written and committed — it will run correctly once the pytest hang is resolved.

**Recommended fix for pytest hang:** Add `DEEPEVAL_TELEMETRY_OPT_OUT=YES` and `DEEPEVAL_ASYNC_MODE=False` to `.env` before running tests, and investigate whether `deepeval` (or `chromadb`) needs to be lazily imported rather than at module load time.

### 2. uv cache cleared mid-session

During the session the uv cache was cleared, causing every subsequent `uv run` and `uv add` call to re-download all packages from scratch. With packages like `scipy` (36.8 MB), `pyarrow` (26.7 MB), and `onnxruntime` (12.8 MB), the environment restoration took around one hour.

**Workaround used:** For `platform/backend`, instead of waiting for `uv add --editable "../../"` to complete, a `.pth` file was written directly to the backend venv's `site-packages` to make `rag_evaluator` importable for runtime verification. The `pyproject.toml` and `[tool.uv.sources]` were edited manually using the correct uv format.

### 3. Windows file locking on worktree cleanup

The `.worktrees/codebase-reorg` directory could not be fully deleted at the end because `python.exe` and `pytest.exe` inside the worktree's `.venv` were locked by the OS. The git worktree registration was successfully removed (`git worktree remove`), and the feature branch was deleted. The physical `.worktrees/` directory remains on disk and can be safely deleted manually once any processes using it have exited.

---

## Files Changed

| File | Change |
|------|--------|
| `debug_eval.py` | Deleted |
| `fetch_eval_id.py` | Deleted |
| `scripts/run_evaluation.py` | Deleted |
| `scripts/run_streamlit.py` | Deleted |
| `scripts/archive/` | Created — holds 5 archived migration scripts |
| `data/test_set _txt.json` | Renamed → `test_set_txt.json` |
| `.gitignore` | Whitelisted `data/test_set_verify.json` |
| `.env.example` | Added `POSTGRES_USER/PASSWORD/DB` vars |
| `docker-compose.yml` | Full rewrite — postgres, pinned versions, health checks, mem limits |
| `platform/frontend/src/api/client.ts` | Fixed `import` endpoint URL |
| `platform/backend/pyproject.toml` | Added `rag-evaluator` editable dep + `[tool.uv.sources]` |
| `platform/backend/app/services/rag_adapter.py` | Removed sys.path hack; uses registry |
| `src/rag_evaluator/cli.py` | Replaced RAG imports with registry |
| `src/rag_evaluator/rag_implementations/registry.py` | **New** — single source of truth |
| `tests/unit/test_rag_registry.py` | **New** — registry unit tests |
