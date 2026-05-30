# Split RAG Configuration into Build-time and Query-time Parameters - v2

**Status:** Proposed
**Date:** 2026-05-30
**Author:** design notes, revised after review

## 1. Problem

Today the platform treats a RAG configuration as one flat blob. When a Knowledge
Base is indexed, the config is frozen into `KnowledgeBaseIndex.config_snapshot`.
Evaluation and Playground then instantiate RAGs from that snapshot, so query-time
experiments are coupled to the config used at index build time.

That causes three concrete problems:

1. A ready index cannot be evaluated with different query-time settings such as
   RLM `orchestrator_model`, generation `llm_model`, RLM budgets, or `top_k`.
2. Attempting to change query-time settings implies rebuilding or re-preparing
   artifacts, even when the stored index does not need to change.
3. The current evaluation path still calls `prepare_documents()` for ready
   indexes. That is not a harmless load operation. Several implementations can
   mutate or rewrite stored artifacts during preparation.

The missing spine is:

```text
Build path:
  RAGConfig + KnowledgeBase -> immutable KnowledgeBaseIndex artifact
  This is the only path allowed to call prepare/build methods.

Query path:
  KnowledgeBaseIndex + query_overrides -> loaded RAG instance
  This path must never call a method that can rewrite the index.
```

## 2. Decisions

### 2.1 Use an explicit load path

Do not rely on idempotent `prepare_documents()` behavior. `prepare_documents()`
means "build or mutate stored artifacts" in the current codebase, so using it
from evaluation or playground is the wrong lifecycle boundary.

Add an explicit query/load path:

- `IndexBuildService.build_index(...)` constructs a build-time RAG and calls
  `prepare_documents(...)`.
- `EvaluationRunner` and `PlaygroundService` call a new load/open path and never
  call `prepare_documents(...)` for ready indexes.
- Implementations that need runtime initialization from stored artifacts expose
  a non-mutating load method.

Proposed service shape:

```python
rag = rag_adapter.create_rag_for_index_build(index)
await rag_adapter.prepare_documents(rag, documents_path)

rag, effective = rag_adapter.load_rag_for_index_query(index, query_overrides)
result = await rag_adapter.query_with_trace(rag, question, top_k=effective.top_k)
```

Implementation notes:

- Vector RAGs may only need constructor-time connection/open logic.
- `filesystem_rag` should initialize its agent from `prepared_path` without
  running the preparation pipeline.
- `rlm_rag` should load the prepared catalog, choose simple vs agent mode, and
  optionally validate the manifest without regenerating files.
- If a ready index cannot be loaded, fail the evaluation with a clear error
  instead of silently rebuilding.

### 2.2 Consolidate the RAG parameter registry

There must be one canonical source of truth for RAG parameter metadata and
build/query phase classification.

Today there are two paths:

- Core: `src/rag_evaluator/rag_implementations/registry.py`
- Platform API/UI: `platform/backend/app/services/rag_registry.py`

The platform API currently serves `/rag-types` from the platform registry, not
from `RAGAdapterService.get_parameter_schema()`. Updating only the core registry
would not reach the frontend.

Decision:

- Core owns canonical RAG type parameter definitions and phase classification.
- Platform registry may remain as an adapter for Pydantic response models and
  LLM provider metadata, but it must be generated from or directly backed by the
  core registry.
- Validation, API schema responses, and frontend forms must consume the same
  parameter metadata.

Each parameter definition should include at least:

```python
{
    "name": "max_repl_steps",
    "type": "integer",
    "phase": "query",  # "build" or "query"
    "default": 15,
    "description": "...",
    "minimum": 1,
    "maximum": 50,
}
```

Platform-managed storage/connection fields such as `qdrant_url`, `neo4j_uri`,
credentials, `prepared_path`, and physical collection/index names should be
classified as build/non-overridable for query validation. They may also carry a
`platform_managed` flag for UI behavior.

### 2.3 Persist embedding and build-model provenance

If a value affects stored artifacts, it must be persisted in the build snapshot.
Do not infer build-critical values from current environment defaults at query
time.

Required changes:

- Add `embedding_model` to platform `RAGConfig` and schemas, defaulting to the
  current embedding setting for backward compatibility.
- Include resolved `embedding_model` in `KnowledgeBaseIndex.config_snapshot`.
- Keep `KnowledgeBaseIndex.embedding_model` as build metadata, but do not rely on
  it as the only source of provenance.
- Persist other environment-derived build values when they shape artifacts, for
  example hybrid sparse model name and graph extraction model.

Graph RAG currently uses top-level `llm_model` during graph indexing and query
generation. The split should introduce or persist a distinct build-time
`extraction_model` for graph construction. For legacy indexes, default
`extraction_model` to the snapshot `llm_model` that was used at build time.

### 2.4 Decouple RAG generation model from judge model

The RAG generation model is part of the system under test. The DeepEval judge
model is part of the measurement apparatus. They should be separately
configurable and separately recorded.

Add:

- `EvaluationCreate.eval_judge_model: str | None`
- `Evaluation.eval_judge_model` or equivalent persisted field
- `RunManifest.eval_judge_model`

Backward compatibility:

- If `eval_judge_model` is omitted, default to the effective RAG generation
  model, preserving current behavior.

Recommended evaluation practice:

- Keep `eval_judge_model` stable across comparison runs.
- Vary `llm_model` or `orchestrator_model` as part of the RAG being tested.

### 2.5 Keep v1's overridable surface small

Do not over-promise. v1 should expose only knobs that the current core
implementations actually support or can support with small, clear changes.

Future-facing knobs such as hybrid fusion weights and graph traversal depth are
valid goals, but they are not v1 unless the core implementation first exposes
them.

## 3. Data Model

### 3.1 Query override shape

Use a structured override shape instead of one flat dict. This avoids mixing
RAG-specific parameters, top-level config fields, and method arguments.

```json
{
  "llm_model": "gpt-5-mini",
  "top_k": 8,
  "parameters": {
    "orchestrator_model": "gpt-5-mini",
    "max_repl_steps": 20
  }
}
```

Semantics:

- `llm_model`: query-time RAG generation/orchestration model.
- `top_k`: method-level retrieval argument, passed to `query()` or
  `query_with_trace()`.
- `parameters`: RAG-type-specific query parameters.
- `eval_judge_model`: not part of `query_overrides`; it belongs to the
  evaluation configuration.

### 3.2 Effective config

Build an explicit effective config before running evaluation or playground
queries.

```text
build_snapshot = index.config_snapshot
query_overrides = evaluation.query_overrides or {}

effective_config = build_snapshot with:
  - query-phase parameter overrides applied
  - query-time llm_model override applied
  - top_k carried separately as query execution metadata
  - build-phase values unchanged
```

Validation rules:

- Reject unknown override keys.
- Reject build-phase parameter overrides with HTTP 400.
- Reject top-level build overrides such as `embedding_model`.
- Error messages must explain that changing a build parameter requires a new
  index.

Example rejection:

```text
Cannot override build-time parameter `embedding_model` for an existing index.
Changing it requires creating a new index.
```

## 4. Initial Parameter Classification

This table is intentionally conservative. "Future" means useful, but not part of
the first implementation unless the core exposes the parameter.

| RAG type | Build-time, frozen in index | Query-time v1 | Future query-time |
|---|---|---|---|
| `vector_semantic` | `embedding_model`, `chunk_size`, `chunk_overlap`, physical collection/storage | `llm_model`, `top_k` | reranker |
| `vector_hybrid` | dense `embedding_model`, sparse model name, `chunk_size`, `chunk_overlap`, physical Qdrant collection/storage | `llm_model`, `top_k` | RRF/fusion tuning after core support exists |
| `graph_rag` | `embedding_model`, graph `extraction_model`, graph schema/build settings, physical labels/vector index/storage | `llm_model`, `top_k` | retrieval depth/hops after core support exists |
| `filesystem_rag` | prepared filesystem, `word_threshold`, preparation options, physical prepared path | `llm_model`, `max_iterations`, `max_tool_calls`, `max_file_reads` | richer agent controls |
| `rlm_rag` | `chunk_size`, `chunk_overlap`, `use_llm_summaries`, `use_llm_topics`, `max_topics_per_doc`, `worker_model`, physical prepared path | `llm_model`, `orchestrator_model`, `security_mode`, `max_repl_steps`, `repl_timeout`, `max_file_reads`, `max_read_bytes`, `max_read_lines`, `max_sub_calls`, `max_recursion_depth`, `small_corpus_threshold`, `top_k` for simple mode | query-time `worker_model` only after per-model artifact/query semantics are explicit |

Notes:

- `worker_model` is mixed in RLM: it shapes stored summaries/topics and is also
  used for query-time sub-LLM calls. v1 freezes it as build-time.
- `top_k` is not a constructor parameter. It must be passed to the query call.
- `filesystem_rag` ignores `top_k`; do not expose it as a meaningful override
  for that type.
- `rlm_rag` ignores `top_k` in agent mode but uses it in simple-context mode.

## 5. Backend Changes

### 5.1 Core library

Files:

- `src/rag_evaluator/rag_implementations/registry.py`
- `src/rag_evaluator/common/base_rag.py`
- RAG implementations under `src/rag_evaluator/rag_implementations/`

Required work:

- Extend the canonical registry with `phase`.
- Add helpers:
  - `get_parameter_schema(rag_type)`
  - `build_param_names(rag_type)`
  - `query_param_names(rag_type)`
  - `validate_query_overrides(rag_type, overrides)`
  - `split_parameters(rag_type, parameters)`
- Add or standardize a non-mutating load/open method. Suggested default:

```python
class BaseRAG:
    def load_index(self) -> None:
        """Open existing prepared/indexed artifacts without mutating them."""
        return None
```

Implementation-specific expectations:

- Vector semantic: constructor opens Chroma collection; `load_index()` can be a
  no-op or validate collection existence/count.
- Vector hybrid: constructor opens Qdrant collection; `load_index()` can validate
  collection existence/count.
- Graph RAG: constructor opens driver/retriever; `load_index()` can validate
  expected labels/index.
- Filesystem RAG: initialize agent from existing prepared path without running
  `PreparationPipeline`.
- RLM RAG: load catalog, choose routing mode, initialize simple/agent mode, and
  do not call `DocumentProcessor.prepare()`.

RLM `_hash_config`:

- Keep it aligned with build-phase registry values.
- Do not include query-time values such as `orchestrator_model` or
  `small_corpus_threshold`.
- Tests should prove changing query overrides does not change the build hash.

### 5.2 Backend models and migration

Files:

- `platform/backend/app/models/rag_config.py`
- `platform/backend/app/models/evaluation.py`
- `platform/backend/app/models/run_manifest.py`
- `platform/backend/app/models/knowledge_base_index.py`
- Alembic migrations

Required columns:

- `rag_configs.embedding_model` with a default/backfill.
- `evaluations.query_overrides` JSON, default `{}`.
- `evaluations.eval_judge_model` nullable string or equivalent persisted field.
- `run_manifests.build_config_snapshot` JSON.
- `run_manifests.query_overrides` JSON, default `{}`.
- `run_manifests.effective_config_snapshot` JSON.
- Keep `run_manifests.rag_config_snapshot` for compatibility. It may mirror the
  effective config for new runs or be marked legacy/deprecated.

Index snapshot:

- Keep `knowledge_base_indexes.config_snapshot`.
- Include resolved build-critical fields:
  - `rag_type`
  - build `parameters`
  - query default `parameters`
  - `llm_provider`
  - default generation `llm_model`
  - `llm_base_url`
  - `embedding_model`
  - graph `extraction_model` where applicable
  - hybrid sparse model name where applicable

Migration notes:

- SQLite migrations need Alembic batch mode.
- Existing evaluations get `query_overrides = {}`.
- Existing run manifests can leave new snapshots as nullable if needed, or
  backfill:
  - `build_config_snapshot = rag_config_snapshot`
  - `effective_config_snapshot = rag_config_snapshot`
  - `query_overrides = {}`
- Existing `eval_judge_model` should default to existing `generation_model` or
  snapshot `llm_model`.

### 5.3 RAG adapter service

Files:

- `platform/backend/app/services/rag_adapter.py`

Required work:

- Replace the overloaded `create_rag_for_index()` usage with explicit methods:
  - `create_rag_for_index_build(index)`
  - `load_rag_for_index_query(index, query_overrides=None)`
- Implement central effective-config construction and validation.
- Apply query-time `parameters` and `llm_model` when constructing the query RAG.
- Return both the loaded RAG and effective execution config.
- Do not apply `top_k` to constructor config. Carry it to query execution.
- If query RAG instances are cached, cache by:
  - index physical ID
  - normalized effective config hash
  - relevant load-time query parameters

`top_k` does not necessarily need to be part of the instance cache key if it is
only passed as a method argument, but including it is acceptable for a simpler
first implementation.

### 5.4 Index build service

Files:

- `platform/backend/app/services/index_build_service.py`

Required work:

- Use `create_rag_for_index_build(index)`.
- This remains the only service that calls `prepare_documents()` for an index.
- Include `rlm_rag` in storage type handling if not already present.
- Persist resolved build-critical fields into `config_snapshot` before the build
  starts.

### 5.5 Evaluation runner

Files:

- `platform/backend/app/services/evaluation_runner.py`

Required work:

- Load ready indexes with `load_rag_for_index_query(...)`.
- Remove `prepare_documents()` from the ready-index evaluation path.
- Extract effective `top_k` and pass it to `query_with_trace(...)`.
- Initialize DeepEval metrics with `evaluation.eval_judge_model`.
- Use the effective RAG generation model for RAG generation cost attribution.
- Do not use the judge model for RAG answer cost attribution.
- Record/emit effective config metadata for observability.

Cost caveat:

- RLM can use both orchestrator and worker models during query. v1 may keep the
  existing approximate cost model, but the manifest must record the effective
  models clearly. A later improvement can add per-model token accounting.

### 5.6 Evaluation API and schemas

Files:

- `platform/backend/app/schemas/evaluation.py`
- `platform/backend/app/api/evaluations.py`

Required work:

- `EvaluationCreate`:
  - `query_overrides: QueryOverrides | None`
  - `eval_judge_model: str | None`
- Validate overrides during `create_evaluation` before persisting.
- Compute effective config once at creation time.
- Persist:
  - `Evaluation.query_overrides`
  - `Evaluation.eval_judge_model`
  - `RunManifest.build_config_snapshot`
  - `RunManifest.query_overrides`
  - `RunManifest.effective_config_snapshot`
  - `RunManifest.generation_model`
  - `RunManifest.eval_judge_model`
- Return override/effective config fields from response schemas where useful.

### 5.7 RAG config API and schemas

Files:

- `platform/backend/app/api/rag_configs.py`
- `platform/backend/app/schemas/rag_config.py`
- `platform/backend/app/services/rag_registry.py`

Required work:

- Add `phase` to `RAGTypeParameter`.
- Add `embedding_model` to `RAGConfigCreate`, `RAGConfigUpdate`, and responses.
- Serve RAG parameter metadata from the consolidated core-backed registry.
- Keep LLM provider metadata in the platform layer.

### 5.8 Playground API and service

Files:

- `platform/backend/app/schemas/playground.py`
- `platform/backend/app/api/playground.py`
- `platform/backend/app/services/playground_service.py`

Required work:

- Add optional `query_overrides`.
- Validate overrides against each selected index.
- Load via the query/load path.
- Pass effective `top_k` to query execution.
- Store effective config metadata with playground query history if useful.

## 6. Frontend Changes

Files:

- `platform/frontend/src/api/client.ts`
- `platform/frontend/src/components/evaluations/StartEvaluationWizard.tsx`
- `platform/frontend/src/components/evaluations/ManifestViewer.tsx`
- `platform/frontend/src/components/rag-configs/RAGConfigDialog.tsx`
- `platform/frontend/src/components/indexes/CreateIndexDialog.tsx`
- `platform/frontend/src/components/knowledge-bases/IndexKBDialog.tsx`
- `platform/frontend/src/components/comparisons/ConfigDiff.tsx`
- `platform/frontend/src/components/comparisons/compare-utils.ts`
- `platform/frontend/src/components/playground/*`

Required work:

- Add `phase` to parameter schema types.
- Add `embedding_model` to RAG config types.
- Add `query_overrides` and `eval_judge_model` to evaluation create types.
- Add manifest fields for build snapshot, overrides, effective config, generation
  model, and judge model.
- In `RAGConfigDialog`, group fields by build/query phase.
- In index creation dialogs, emphasize build fields and mark query fields as
  defaults that can be changed at evaluation time.
- In `StartEvaluationWizard`, add a query overrides step:
  - Show frozen build params read-only.
  - Allow editing only query params.
  - Include `top_k`.
  - Include RAG generation model override.
  - Include separate judge model selection.
- In `ManifestViewer`, show:
  - build snapshot
  - query overrides
  - effective RAG config
  - generation model
  - judge model
- In comparison views, diff effective configs and flag build differences as
  requiring separate indexes.

v1 UI should stay honest: if a parameter is future-only, do not render it as an
editable override.

## 7. Tests

### 7.1 Registry

- Every exposed RAG parameter has a phase.
- Build/query partitions are exhaustive and disjoint.
- Backend `/rag-types` phase data matches the core registry.

### 7.2 Override validation

- Query param override succeeds.
- Build param override returns HTTP 400.
- Unknown override returns HTTP 400.
- `llm_model` override changes only generation/orchestration.
- `embedding_model` override is rejected.
- `top_k` is accepted and carried as query execution metadata.

### 7.3 Load path

- Evaluation of a ready index does not call `prepare_documents()`.
- Playground query of a ready index does not call `prepare_documents()`.
- Filesystem RAG loads an existing prepared directory without running the
  preparation pipeline.
- RLM RAG loads an existing prepared directory without changing manifest hash.

### 7.4 Evaluation plumbing

- `query_overrides` are persisted on `Evaluation`.
- `eval_judge_model` is persisted and used for DeepEval metrics.
- Run manifest records build snapshot, overrides, effective config, generation
  model, and judge model.
- Effective `top_k` reaches `query_with_trace(...)`.
- RAG answer cost uses the effective generation model, not the judge model.

### 7.5 RLM end-to-end

- Build one RLM index with worker/build config A.
- Run two evaluations with different `orchestrator_model`.
- Assert no re-preparation.
- Assert manifest/config hash for build artifacts is unchanged.
- Assert effective configs differ only in query-time fields.

## 8. Incremental Rollout

1. Consolidate registry and add phase metadata. Add tests proving backend API
   output comes from the same source.
2. Persist build-critical model provenance, especially `embedding_model`.
3. Add explicit non-mutating load path for all RAG types. Remove evaluation-time
   `prepare_documents()` for ready indexes.
4. Add query override schema, validation, and effective config helper.
5. Add evaluation plumbing: `query_overrides`, `eval_judge_model`, manifest
   snapshots, effective `top_k`, and cost attribution updates.
6. Ship v1 frontend controls for the small supported override set.
7. Add Playground override support.
8. Later: expose hybrid fusion tuning, graph depth/hops, richer filesystem
   budgets, and possibly query-time RLM `worker_model` after core support and
   provenance semantics are explicit.

## 9. Definition of Done

- Index build and index query are separate lifecycle paths.
- Evaluation and Playground never call build/preparation methods for ready
  indexes.
- Query overrides can change allowed query-time settings without rebuilding.
- Build-time override attempts are rejected with clear messages.
- `top_k` is actually passed to query execution.
- The platform API and frontend consume the same phase metadata that validation
  uses.
- `embedding_model` and other build-critical model choices are persisted.
- RAG generation model and DeepEval judge model are decoupled and recorded.
- Run manifest records build snapshot, query overrides, effective config,
  generation model, and judge model.
- Existing evaluations and indexes continue to work with compatibility defaults.
