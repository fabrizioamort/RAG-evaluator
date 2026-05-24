# Codebase Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up organic growth — remove dead code, fix a frontend bug, improve Docker config, and replace the sys.path hack with a proper package dependency.

**Architecture:** The project has two packages: `src/rag_evaluator` (CLI) and `platform/backend` (web platform). The platform already imports from the CLI package via a `sys.path.insert` hack in `rag_adapter.py`. This plan formalizes that relationship, cleans up dead scripts, fixes a known bug in `client.ts`, and hardens Docker Compose. The evaluation runner duplication (the largest issue) is left for a separate sprint — it carries too much platform-specific logic (checkpointing, cancel/pause, webhooks) to refactor safely here.

**Tech Stack:** Python/uv, FastAPI, React/TypeScript, Docker Compose

---

## Phase 1: Safe Cleanup

### Task 1: Delete dead root-level files

No tests needed — these files are not imported by anything.

**Files:**
- Delete: `debug_eval.py`
- Delete: `fetch_eval_id.py`
- Delete: `nul`

**Step 1: Verify nothing imports these files**

Run:
```powershell
cd C:\Users\fabri\projects\RAG-evaluator
grep -r "debug_eval\|fetch_eval_id" src/ platform/ scripts/ tests/ 2>$null
```
Expected: no output (confirming they are not imported anywhere)

**Step 2: Delete the files**

```powershell
Remove-Item debug_eval.py, fetch_eval_id.py, nul
```

**Step 3: Verify deletion**

```powershell
ls debug_eval.py 2>$null; echo "done"
```
Expected: error or "done" only — file should not exist.

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete dead root-level debug scripts and nul artifact"
```

---

### Task 2: Archive one-off migration scripts

These 5 scripts were one-time data migrations. They should not be in the active `scripts/` directory.

**Files:**
- Create: `scripts/archive/` directory
- Move: `scripts/fix_legacy_metrics.py` → `scripts/archive/`
- Move: `scripts/fix_migration_data.py` → `scripts/archive/`
- Move: `scripts/fix_missing_answers.py` → `scripts/archive/`
- Move: `scripts/migrate_existing_evaluations.py` → `scripts/archive/`
- Move: `scripts/migrate_legacy_data.py` → `scripts/archive/`
- Delete: `scripts/run_streamlit.py` (redundant — `rag-eval ui` does the same)
- Delete: `scripts/run_evaluation.py` (redundant — `rag-eval evaluate` does the same)

**Step 1: Create archive dir and move scripts**

```powershell
New-Item -ItemType Directory -Path scripts\archive -Force
Move-Item scripts\fix_legacy_metrics.py scripts\archive\
Move-Item scripts\fix_migration_data.py scripts\archive\
Move-Item scripts\fix_missing_answers.py scripts\archive\
Move-Item scripts\migrate_existing_evaluations.py scripts\archive\
Move-Item scripts\migrate_legacy_data.py scripts\archive\
Remove-Item scripts\run_streamlit.py
Remove-Item scripts\run_evaluation.py
```

**Step 2: Verify active scripts still present**

```powershell
ls scripts\
```
Expected output includes: `load_multihop_rag.py`, `verify_pipeline.py`, `test_chroma_rag.py`, `clean_data.py`, `inspect_db.py`, `archive\`

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: archive completed migration scripts, remove redundant runners"
```

---

### Task 3: Fix test set filenames with spaces

Two data files have spaces in their names — likely accidental. Fix them and update `.gitignore` to exclude all non-canonical test sets.

**Files:**
- Rename: `data/test_set RAG_5.json` → `data/test_set_rag5.json`
- Rename: `data/test_set _txt.json` → `data/test_set_txt.json`
- Modify: `.gitignore`

**Step 1: Rename files**

```powershell
Rename-Item "data\test_set RAG_5.json" "test_set_rag5.json"
Rename-Item "data\test_set _txt.json" "test_set_txt.json"
```

**Step 2: Update .gitignore to track only canonical test sets**

Current `.gitignore` has:
```
data/*.json
!data/test_set.json
```

Replace with a list that keeps all known test sets (they are small, useful to version):
```
# Test sets (keep all, exclude nothing here)
```

Actually, looking at the current `.gitignore`, the rule `data/*.json` + `!data/test_set.json` means only `test_set.json` is committed. The multihop, custom, verify, rag5, and txt variants are all gitignored.

The fix: either explicitly whitelist the others too, or leave them gitignored (they can be regenerated with scripts). The cleanest option: keep current rule but also whitelist `test_set_verify.json` since it's used by `verify_pipeline.py`.

Edit `.gitignore` — find the data section:
```
# Data directories
data/raw/*
data/processed/*
data/prepared/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/prepared/.gitkeep
data/*.json
!data/test_set.json
```

Change to:
```
# Data directories
data/raw/*
data/processed/*
data/prepared/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/prepared/.gitkeep
data/*.json
!data/test_set.json
!data/test_set_verify.json
```

**Step 3: Check git status — confirm renamed files and gitignore change**

```bash
git status
```

**Step 4: Commit**

```bash
git add data/test_set_rag5.json data/test_set_txt.json .gitignore
git commit -m "chore: fix test set filenames with spaces, whitelist verify test set in gitignore"
```

---

## Phase 2: Bug Fix

### Task 4: Fix client.ts `import()` wrong URL

**Context:** The backend's import endpoint is `POST /projects/{project_id}/test-sets/import` (line 459 in `platform/backend/app/api/test_sets.py`). But `client.ts` line 656 calls `POST /test-sets/${id}/import` — wrong path structure, wrong parameter.

**Files:**
- Modify: `platform/frontend/src/api/client.ts:656`

**Step 1: Read the backend endpoint signature to confirm correct URL**

From `platform/backend/app/api/test_sets.py`:
```python
@router.post("/projects/{project_id}/test-sets/import", ...)
async def import_test_set(project_id: UUID, import_data: TestSetImport, ...):
```

The correct call is: `POST /projects/{project_id}/test-sets/import` with the full import payload (including name, description, test cases).

**Step 2: Fix client.ts**

Current (line 656):
```typescript
import: (id: string, data: unknown) => apiClient.post(`/test-sets/${id}/import`, data),
```

Replace with:
```typescript
import: (projectId: string, data: unknown) => apiClient.post(`/projects/${projectId}/test-sets/import`, data),
```

Use the Edit tool on `platform/frontend/src/api/client.ts`.

**Step 3: Search for all call sites of `api.testSets.import` in the frontend**

```powershell
grep -r "testSets.import\|testSets\.import" platform\frontend\src\ 2>$null
```

For each call site found, update the argument from `testSetId` to `projectId`. The projectId is available from the current project context wherever this is called.

**Step 4: Run frontend lint**

```powershell
cd platform\frontend
npm run lint
```
Expected: no errors related to the changed function signature.

**Step 5: Commit**

```bash
cd ../..
git add platform/frontend/src/api/client.ts
git commit -m "fix: correct import test set API client URL to use project-scoped endpoint"
```

---

## Phase 3: Docker Compose Hardening

### Task 5: Add PostgreSQL, memory limits, and health checks to docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Read the current docker-compose.yml**

```powershell
cat docker-compose.yml
```

**Step 2: Verify the platform backend DATABASE_URL default**

```powershell
grep -n "DATABASE_URL\|postgres\|sqlite" platform\backend\app\config.py
```

Note the default. If it defaults to SQLite for dev, we still want PostgreSQL available as an option.

**Step 3: Update docker-compose.yml**

Replace the entire file content with:

```yaml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-rageval}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-rageval}
      POSTGRES_DB: ${POSTGRES_DB:-rageval}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    mem_limit: 512m
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-rageval}"]
      interval: 10s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:v1.8.4
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    mem_limit: 2g
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:6333/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:5.18-community
    environment:
      NEO4J_AUTH: ${NEO4J_AUTH:-neo4j/password}
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    mem_limit: 2g
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p ${NEO4J_PASSWORD:-password} 'RETURN 1' || exit 1"]
      interval: 20s
      timeout: 10s
      retries: 5
      start_period: 30s

volumes:
  postgres_data:
  qdrant_data:
  neo4j_data:
```

**Step 4: Update .env.example with new Postgres vars**

```powershell
cat .env.example
```

Add to `.env.example` if not present:
```
# Docker Compose - PostgreSQL
POSTGRES_USER=rageval
POSTGRES_PASSWORD=rageval
POSTGRES_DB=rageval
```

**Step 5: Verify compose file is valid**

```powershell
docker compose config
```
Expected: parsed config printed with no errors.

**Step 6: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "chore: add postgres, pin image versions, add memory limits and health checks to docker-compose"
```

---

## Phase 4: Fix the sys.path Hack

### Task 6: Make `rag-evaluator` a proper local dependency of the platform backend

**Context:** `platform/backend/app/services/rag_adapter.py` lines 19-22 do:
```python
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
```
This is fragile (path traversal depth, not declared as dependency). Replace it with a proper `uv` path dependency.

**Files:**
- Modify: `platform/backend/pyproject.toml`
- Modify: `platform/backend/app/services/rag_adapter.py` (remove sys.path hack)

**Step 1: Add `rag-evaluator` as a local editable dependency**

```powershell
cd platform\backend
uv add --editable "../../"
```

This adds `rag-evaluator @ file:///...` to `platform/backend/pyproject.toml` and installs the package into the platform's `.venv`.

**Step 2: Verify the package is importable**

```powershell
uv run python -c "from rag_evaluator.common.base_rag import BaseRAG; print('OK')"
```
Expected: `OK`

**Step 3: Remove the sys.path hack from rag_adapter.py**

In `platform/backend/app/services/rag_adapter.py`, remove lines 8-22 (the `import sys`, `from pathlib import Path`, and the sys.path block):

```python
# REMOVE these lines:
import sys
from pathlib import Path
...
# Add src to path for importing rag_evaluator
src_path = Path(__file__).parent.parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
```

The imports that follow (`from rag_evaluator.common.base_rag import BaseRAG, RAGConfig`) stay unchanged — they now work via the proper package dependency.

Note: `pathlib.Path` is still used later in the file (`str(Path(settings.STORAGE_PATH) / ...)`), so only remove the `sys.path` block, not the `Path` import if it's still needed elsewhere. Check carefully with Read before editing.

**Step 4: Run backend tests to confirm no import errors**

```powershell
cd platform\backend
uv run pytest tests/ -x -q 2>&1 | head -30
```
Expected: tests pass (or fail for unrelated reasons — the key check is no `ImportError` for `rag_evaluator`).

**Step 5: Verify the app starts**

```powershell
uv run uvicorn app.main:app --port 8001 --timeout-graceful-shutdown 1 &
Start-Sleep -Seconds 3
Invoke-WebRequest http://localhost:8001/api/v1/health -UseBasicParsing | Select-Object StatusCode
# kill the server
Stop-Process -Name "uvicorn" -ErrorAction SilentlyContinue
```
Expected: `StatusCode: 200`

**Step 6: Commit**

```bash
cd ../..
git add platform/backend/pyproject.toml platform/backend/uv.lock platform/backend/app/services/rag_adapter.py
git commit -m "refactor: replace sys.path hack with proper uv local package dependency for rag-evaluator"
```

---

## Phase 5: Consolidate RAG Type Registry

### Task 7: Move RAG type registry to `src/` as the single source of truth

**Context:** `rag_adapter.py` defines `RAG_TYPE_REGISTRY` (mapping type strings to import paths) and `RAG_TYPE_PARAMETERS` (parameter schemas). The CLI's `cli.py` duplicates the same 4 types in `get_rag_implementation()`. These should live in one place.

**Files:**
- Create: `src/rag_evaluator/rag_implementations/registry.py`
- Modify: `src/rag_evaluator/cli.py` (use registry)
- Modify: `platform/backend/app/services/rag_adapter.py` (use registry)

**Step 1: Write a test for the new registry module**

Create `tests/unit/test_rag_registry.py`:
```python
from rag_evaluator.rag_implementations.registry import RAG_TYPES, get_rag_class


def test_all_types_present():
    assert set(RAG_TYPES) == {"vector_semantic", "vector_hybrid", "graph_rag", "filesystem_rag"}


def test_get_rag_class_returns_type():
    from rag_evaluator.common.base_rag import BaseRAG
    cls = get_rag_class("vector_semantic")
    assert issubclass(cls, BaseRAG)


def test_get_rag_class_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown RAG type"):
        get_rag_class("nonexistent")
```

**Step 2: Run the test — verify it fails**

```powershell
cd C:\Users\fabri\projects\RAG-evaluator
uv run pytest tests/unit/test_rag_registry.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` — registry module does not exist yet.

**Step 3: Create `src/rag_evaluator/rag_implementations/registry.py`**

```python
"""Registry mapping RAG type keys to implementation classes and parameter schemas."""

import importlib
from typing import Any

from rag_evaluator.common.base_rag import BaseRAG

# Maps RAG type key -> fully-qualified class path
RAG_CLASS_PATHS: dict[str, str] = {
    "vector_semantic": "rag_evaluator.rag_implementations.vector_semantic.chroma_rag.ChromaSemanticRAG",
    "vector_hybrid": "rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag.HybridSearchRAG",
    "graph_rag": "rag_evaluator.rag_implementations.graph_rag.neo4j_rag.Neo4jGraphRAG",
    "filesystem_rag": "rag_evaluator.rag_implementations.filesystem_rag.filesystem_rag.FilesystemRAG",
}

# Human-readable metadata for each type
RAG_TYPES: dict[str, dict[str, str]] = {
    "vector_semantic": {
        "name": "Vector Semantic Search",
        "description": "ChromaDB-based semantic vector search using embeddings",
    },
    "vector_hybrid": {
        "name": "Hybrid Search",
        "description": "Qdrant-based hybrid search combining dense and sparse vectors with RRF fusion",
    },
    "graph_rag": {
        "name": "Graph RAG",
        "description": "Neo4j-based graph RAG with entity relationships and vector search",
    },
    "filesystem_rag": {
        "name": "Filesystem RAG",
        "description": "LLM-guided agent that navigates a prepared filesystem structure",
    },
}

# Parameter schemas for each type (used by the platform UI)
RAG_TYPE_PARAMETERS: dict[str, dict[str, Any]] = {
    "vector_semantic": {
        "properties": {
            "chunk_size": {"type": "integer", "default": 1000, "description": "Size of text chunks"},
            "chunk_overlap": {"type": "integer", "default": 200, "description": "Overlap between chunks"},
            "collection_name": {"type": "string", "default": "rag_documents", "description": "ChromaDB collection name"},
        },
    },
    "vector_hybrid": {
        "properties": {
            "chunk_size": {"type": "integer", "default": 500, "description": "Size of text chunks"},
            "chunk_overlap": {"type": "integer", "default": 50, "description": "Overlap between chunks"},
            "collection_name": {"type": "string", "default": "hybrid_rag", "description": "Qdrant collection name"},
        },
    },
    "graph_rag": {
        "properties": {
            "vector_index_name": {"type": "string", "default": "chunk_embeddings", "description": "Neo4j vector index name"},
        },
    },
    "filesystem_rag": {
        "properties": {
            "word_threshold": {"type": "integer", "default": 1000, "description": "Word count threshold"},
            "max_iterations": {"type": "integer", "default": 10, "description": "Max ReAct loop iterations"},
            "max_tool_calls": {"type": "integer", "default": 20, "description": "Max tool calls per query"},
            "max_file_reads": {"type": "integer", "default": 10, "description": "Max file reads per query"},
        },
    },
}


def get_rag_class(rag_type: str) -> type[BaseRAG]:
    """Return the RAG implementation class for the given type key.

    Raises:
        ValueError: If rag_type is not in the registry.
        ImportError: If the class cannot be imported.
    """
    if rag_type not in RAG_CLASS_PATHS:
        raise ValueError(f"Unknown RAG type: {rag_type}. Supported: {list(RAG_CLASS_PATHS)}")
    module_path, class_name = RAG_CLASS_PATHS[rag_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[return-value]
```

**Step 4: Run the test — verify it passes**

```powershell
uv run pytest tests/unit/test_rag_registry.py -v
```
Expected: all 3 tests PASS.

**Step 5: Update `cli.py` to use the registry**

In `src/rag_evaluator/cli.py`, replace `get_rag_implementation`:

Remove the 4 direct imports at the top:
```python
from rag_evaluator.rag_implementations.filesystem_rag.filesystem_rag import FilesystemRAG
from rag_evaluator.rag_implementations.graph_rag import Neo4jGraphRAG
from rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag import HybridSearchRAG
from rag_evaluator.rag_implementations.vector_semantic.chroma_rag import ChromaSemanticRAG
```

Add one import:
```python
from rag_evaluator.rag_implementations.registry import RAG_TYPES, get_rag_class
```

Replace `get_rag_implementation()`:
```python
def get_rag_implementation(rag_type: str) -> BaseRAG:
    """Get RAG implementation instance by type."""
    return get_rag_class(rag_type)()
```

Also update the argparse `choices` list to use `list(RAG_TYPES)` instead of the hardcoded list:
```python
choices=list(RAG_TYPES) + ["all"],
```

**Step 6: Update `rag_adapter.py` to use the registry**

In `platform/backend/app/services/rag_adapter.py`:

Add import at top:
```python
from rag_evaluator.rag_implementations.registry import (
    RAG_TYPES,
    RAG_TYPE_PARAMETERS,
    get_rag_class,
)
```

Remove the local `RAG_TYPE_REGISTRY` and `RAG_TYPE_PARAMETERS` dicts (they are now in src/).

Replace `_get_rag_class()` method body with:
```python
def _get_rag_class(self, rag_type: str) -> type[BaseRAG]:
    return get_rag_class(rag_type)
```

Replace `get_available_rag_types()` method body with:
```python
def get_available_rag_types(self) -> list[dict[str, Any]]:
    return [
        {"type": k, "name": v["name"], "description": v["description"]}
        for k, v in RAG_TYPES.items()
    ]
```

Replace `get_parameter_schema()` body with:
```python
def get_parameter_schema(self, rag_type: str) -> dict[str, Any]:
    if rag_type not in RAG_TYPE_PARAMETERS:
        raise ValueError(f"Unknown RAG type: {rag_type}")
    return RAG_TYPE_PARAMETERS[rag_type]
```

**Step 7: Run all tests**

```powershell
# CLI/core tests
uv run pytest tests/unit/ -v -x 2>&1 | tail -20

# Platform backend tests
cd platform\backend
uv run pytest tests/ -x -q 2>&1 | tail -20
```
Expected: all pass (or pre-existing failures only — no new failures from this change).

**Step 8: Commit**

```bash
cd ../..
git add src/rag_evaluator/rag_implementations/registry.py
git add src/rag_evaluator/cli.py
git add platform/backend/app/services/rag_adapter.py
git add tests/unit/test_rag_registry.py
git commit -m "refactor: extract RAG type registry to src as single source of truth"
```

---

## Summary

| Task | Risk | Files Touched |
|------|------|--------------|
| 1: Delete dead root files | None | 3 files deleted |
| 2: Archive migration scripts | None | 7 files moved/deleted |
| 3: Fix test set filenames | None | 2 renames, .gitignore |
| 4: Fix client.ts import URL | Low | client.ts |
| 5: Docker Compose | Low | docker-compose.yml, .env.example |
| 6: Fix sys.path hack | Medium | pyproject.toml, rag_adapter.py |
| 7: Consolidate RAG registry | Medium | new registry.py, cli.py, rag_adapter.py |

**Not in this plan (deferred):**
- Evaluation runner deduplication (EvaluationRunner vs RAGEvaluator) — too much platform-specific logic; needs separate sprint
- Token tracking consolidation — low priority, not causing bugs
- Test set consolidation (multihop/custom/verify documentation) — non-code, low priority
- Streamlit UI removal (src/rag_evaluator/ui/) — not breaking anything

**Verification after all tasks:**
```powershell
uv run pytest tests/ -q
cd platform\backend && uv run pytest tests/ -q
cd ..\..\platform\frontend && npm run lint
```
