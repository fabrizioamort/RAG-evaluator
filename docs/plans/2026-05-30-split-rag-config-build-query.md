# Split RAG Configuration into Build-time and Query-time Parameters

**Status:** Proposed
**Date:** 2026-05-30
**Author:** design notes

## 1. Problem

Today a RAG configuration is a single flat blob. When a Knowledge Base is indexed,
the *entire* config is frozen into `KnowledgeBaseIndex.config_snapshot`
(`index_build_service.create_index`), and every downstream consumer
(`rag_adapter.create_rag_for_index`) rebuilds the RAG from that frozen snapshot.
Evaluation and Playground both go through `create_rag_for_index`, so:

- **You cannot evaluate an index with a different configuration than the one it was built with.** The `Evaluation` row still carries a `rag_config_id`, but it is ignored whenever an index is attached (`evaluation_runner.run`).
- To try a different *query-time* setting (e.g. a stronger RLM `orchestrator_model`, a different `top_k`, different hybrid fusion weights) you must **build a whole new index**, paying the full embedding / preparation cost again — even though none of those settings affect what is stored.

This is the wrong default for an *evaluation* tool, whose core job is to sweep
settings against a fixed corpus.

## 2. Why splitting the config in two is the right fix

RAG parameters fall into two fundamentally different classes:

- **Build-time parameters** are baked into the stored artifact. Changing them is meaningless without re-indexing, because the stored data would no longer match. The hard wall is the **embedding model** (and chunking): vectors are computed in a specific model's space and cannot be queried by another. For graph RAG the stored graph is shaped by the extraction model; for filesystem/RLM the prepared files are shaped by chunking + the worker model used for summaries/topics.
- **Query-time parameters** never touch what is stored. They only affect *how* a fixed index is queried: generation model, `top_k`, reranking, fusion weights, agent budgets, RLM orchestrator model, etc.

Splitting the config along this line gives us:

1. **Cheap experimentation** — sweep query parameters against one index instead of rebuilding N times.
2. **Correctness by construction** — the system can *reject* an attempt to "evaluate with a different embedding model", which is silently invalid today.
3. **Honest comparisons** — when two evaluations differ only in a query parameter, that is the only thing that differs; build provenance is identical and provable.
4. **Reproducibility preserved** — the index stays immutable; the evaluation records the *effective* config (frozen build snapshot + applied query overrides).

Crucially, the codebase **already encodes this split implicitly** for RLM:
`ManifestManager._hash_config` (`rlm_rag/preparation.py`) only hashes
`chunk_size, chunk_overlap, use_llm_summaries, use_llm_topics, max_topics_per_doc, worker_model`.
Everything else is, by the cache's own logic, build-irrelevant — i.e. query-time.
The split formalizes what the code already believes.

## 3. Conceptual model

A RAG type's parameters are classified as `build` or `query`. The index freezes
build params; an evaluation (or playground run) may supply **query overrides**
that are merged on top of the frozen build snapshot at query time. Overriding a
build param is rejected.

### Per-type classification (initial)

| RAG type | Build-time (frozen in index) | Query-time (overridable) | Override value |
|---|---|---|---|
| `vector_semantic` | `embedding_model`, `chunk_size`, `chunk_overlap`, `collection_name` | `llm_model` (generation), `top_k` | Low |
| `vector_hybrid` | dense + sparse (SPLADE) models, `chunk_size`, `chunk_overlap`, `collection_name` | `top_k`, RRF `k` / fusion weights, `llm_model` | Medium |
| `graph_rag` | extraction model, graph build, `vector_index_name`, embeddings | retrieval depth / hops, `top_k`, `llm_model` | Medium |
| `filesystem_rag` | prepared filesystem (analyzer/synthesizer, chunking) | `max_iterations`, `max_tool_calls`, `max_file_reads`, `llm_model` (verify `word_threshold`) | High |
| `rlm_rag` | `chunk_size`, `chunk_overlap`, `use_llm_summaries`, `use_llm_topics`, `max_topics_per_doc`, `worker_model`* | `orchestrator_model`, `security_mode`, `max_repl_steps`, `repl_timeout`, `max_file_reads`, `max_read_bytes`, `max_read_lines`, `max_sub_calls`, `max_recursion_depth`, `small_corpus_threshold` | Highest |

\* `worker_model` is **mixed**: it shapes stored summaries/topics (build) but is also used for query-time sub-LLM calls. Treat it as **build** for the frozen snapshot; document that overriding it at query time would only change sub-calls, not the stored artifacts.

Top-level `RAGConfig.embedding_model` is always build. `llm_model` is query
(generation/orchestrator). `top_k` is a query param passed to `query()` and is
not currently in `parameters` — the override channel should carry it too.

> The feature pays off most for `vector_hybrid`, `filesystem_rag`, and especially
> `rlm_rag`. For `vector_semantic` the overridable surface is tiny; that is expected and fine.

## 4. Chosen approach

**Registry-driven classification + evaluation-time query overrides.** Keep the
flat `parameters` dict, annotate each parameter as build/query in the registry
(single source of truth), and add an override channel on the evaluation. This is
backward compatible and incremental.

Rejected alternative — *structurally* splitting `RAGConfig.parameters` into
`build_parameters` / `query_parameters` columns: cleaner long-term UX but forces
a data migration of every existing config and a bigger frontend rewrite. Can be a
later enhancement once the classification proves out.

### Merge semantics (the heart of the change)

`create_rag_for_index(index, query_overrides=None)`:

1. Start from `index.config_snapshot["parameters"]` (build authoritative; any query params in the snapshot act as **defaults**).
2. Validate every key in `query_overrides` is classified `query` for `index.config_snapshot["rag_type"]`; reject build keys (HTTP 400) with a clear message.
3. `effective_parameters = {**snapshot_parameters, **validated_query_overrides}`.
4. Allow `llm_model` override (generation/orchestrator) the same way.
5. Build the RAG from `effective_parameters`.

Because `_hash_config` ignores query params, applying query overrides never
invalidates the prepared/embedded index — re-preparation is correctly skipped.
This is the proof the design is safe.

## 5. Impacted areas (complete)

### 5.1 Core library (`src/rag_evaluator/`)
- **`rag_implementations/registry.py`** — extend each entry in `RAG_TYPE_PARAMETERS` with a `"phase": "build" | "query"` flag per property (plus mark top-level `embedding_model`=build, `llm_model`=query). Add helpers `build_param_names(rag_type)` / `query_param_names(rag_type)`.
- **`rag_implementations/rlm_rag/preparation.py`** — `_hash_config` should derive build params from the registry classification (or stay aligned with it) so there is a single source of truth, not two hand-maintained lists.
- **Per-implementation review** — confirm each constructor applies query params at query time and tolerates them changing between build and query. RLM `orchestrator_model` already does; verify vector/hybrid/graph/filesystem read `top_k`/generation model per query rather than caching at init.
- **(Optional) `common/base_rag.py`** — a small `split_parameters()` utility used by both core CLI and platform.

### 5.2 Backend models + migration (`platform/backend/app/models`, `alembic/`)
- **`models/evaluation.py`** — add `query_overrides: dict` (JSON, default `{}`) and optionally `effective_config_snapshot: dict` for fast reads.
- **`models/run_manifest.py`** — extend to record `build_config_snapshot`, `query_overrides`, and `effective_config_snapshot` (today it only stores `rag_config_snapshot`). This is what makes comparisons honest.
- **`models/knowledge_base_index.py`** — no schema change required (snapshot stays full); optionally document that only build params are authoritative. `embedding_model` column already exists.
- **Alembic migration** — add the new columns. **SQLite needs batch mode** (per `CLAUDE.md`). Backfill: existing evaluations get `query_overrides = {}` → behavior identical to today.

### 5.3 Backend services (`platform/backend/app/services`)
- **`rag_adapter.py`** — central change:
  - `create_rag_for_index(index, query_overrides=None)`: implement the merge + validation (§4).
  - `get_parameter_schema(rag_type)`: include the build/query `phase` so the UI can render two groups.
  - `get_or_create_rag` **cache key** must include the override set (currently keyed on `config_model.id` only) — otherwise two evals with different overrides would collide.
- **`index_build_service.py`** — `create_index` may keep storing the full config as `config_snapshot`; the build itself only needs build params (query params are inert here). No behavioral change required, but document intent.
- **`evaluation_runner.py`** — pass `self.evaluation.query_overrides` into `create_rag_for_index`; derive the generation model for **cost** (`llm_model_for_cost`, ~line 452) and `llm_model` (~line 277) from the **effective** config, not the raw snapshot, so cost is attributed to the model actually used.
- **`playground_service.py`** — `_execute_single_query` should accept optional query overrides too; the Playground is the natural place to try query params interactively before committing an evaluation.

### 5.4 Backend API + schemas (`platform/backend/app/api`, `app/schemas`)
- **`schemas/evaluation.py`** — `EvaluationCreate`: add `query_overrides: dict | None`. `EvaluationResponse` / `RunManifestResponse`: expose `effective_config_snapshot` (and overrides) for display.
- **`api/evaluations.py`** (`create_evaluation`) — validate overrides against the index's rag_type (reject build keys), compute the effective config, persist `query_overrides` on the `Evaluation`, and store build snapshot + overrides + effective config in the `RunManifest` (currently lines 152–170).
- **`api/rag_configs.py`** / **`schemas/rag_config.py`** — surface the build/query classification in the parameter-schema endpoint the UI consumes.
- **`api/playground.py`** / **`schemas/playground.py`** — optional `query_overrides` on playground requests.
- **`api/indexes.py`** / **`schemas/knowledge_base_index.py`** — optionally expose which params are frozen (build) so the UI can show them read-only.

### 5.5 Frontend (`platform/frontend/src`)
- **`api/client.ts`** — types: parameter-schema entries gain `phase`; `EvaluationCreate` gains `query_overrides`; manifest/evaluation responses gain `effective_config_snapshot`.
- **`components/evaluations/StartEvaluationWizard.tsx`** — new optional step: show the index's frozen **build** params read-only, and editable **query** overrides (pre-filled from the snapshot defaults). This is the primary UX win.
- **`components/rag-configs/RAGConfigDialog.tsx`** — group the form into "Build" and "Query" sections (driven by the schema `phase`) so authors understand which settings lock the index.
- **`components/indexes/CreateIndexDialog.tsx`** & **`components/knowledge-bases/IndexKBDialog.tsx`** — at index creation, emphasize build params; query params can be hidden or labeled "set at evaluation time".
- **`components/evaluations/ManifestViewer.tsx`** — display the effective config (build snapshot + applied overrides).
- **`components/comparisons/ConfigDiff.tsx`** & `compare-utils.ts` — diff **effective** configs and visually distinguish build vs query differences (so a comparison makes clear whether two runs differ in a rebuild-requiring way or just a query knob).
- **`components/playground/*`** — optional query-override controls mirroring the wizard.

### 5.6 Tests
- **Registry**: classification present and complete for every type; build+query partition is exhaustive and disjoint.
- **`tests/test_services/test_rag_adapter.py`**: merge applies query overrides; rejects build-param overrides; cache key includes overrides.
- **`tests/test_api/test_evaluations.py`**: create eval with `query_overrides`; manifest records effective config; build-param override → 400.
- **RLM end-to-end**: same index, two evals with different `orchestrator_model`, no re-preparation triggered (assert manifest/`_hash_config` unchanged).
- Note: per project memory, `uv run pytest` can hang on this machine — validate with direct `python.exe` imports where needed.

### 5.7 Docs
- `docs/api.md` (evaluation create + manifest fields), `docs/cli.md` (if the CLI exposes query overrides), and a short note in `CLAUDE.md` about the build/query split as the canonical mental model.

## 6. Backward compatibility & migration
- `query_overrides` defaults to `{}` ⇒ existing behavior is byte-identical.
- `config_snapshot` is unchanged; existing indexes need no backfill.
- The classification lives in code (registry), so no data migration for it.
- Only additive DB columns (evaluation + run_manifest), applied via SQLite batch-mode Alembic migration.

## 7. Risks & edge cases
- **Instance cache collision** — `get_or_create_rag` keyed only on config id; must fold overrides into the key.
- **Cost attribution** — must use the effective (overridden) generation model, not the snapshot's.
- **Comparison integrity** — UI must show effective config; comparing across a build-param difference is apples-to-oranges and should be flagged.
- **`worker_model` ambiguity (RLM)** — frozen as build; if ever allowed as a query override, document that stored summaries/topics still reflect build-time worker.
- **`top_k`** — lives outside `parameters`; route it through the same override channel for consistency.
- **Validation messages** — rejecting a build-param override must say *why* ("changing `embedding_model` requires re-indexing").

## 8. Incremental rollout
1. **Registry classification** + helpers (core, no behavior change). Ship + test.
2. **`create_rag_for_index(query_overrides=...)`** merge + validation + cache-key fix (service layer). Unit tests.
3. **Evaluation plumbing** — schema field, `create_evaluation`, `evaluation_runner`, run_manifest, migration. Pilot with **RLM `orchestrator_model`** (highest value, lowest risk; matches the real need of evaluating a gpt-5-nano-built index with a gpt-5-mini orchestrator). End-to-end test: no re-preparation.
4. **Frontend** — StartEvaluationWizard query-override step + ManifestViewer effective config.
5. **Generalize** — hybrid fusion / `top_k`, filesystem agent budgets, graph retrieval depth; Playground overrides; ConfigDiff build/query awareness.
6. **(Optional, later)** structural split of `parameters` into `build`/`query` sub-objects + RAGConfigDialog grouping.

## 9. Definition of done
- An index built with config A can be evaluated with query overrides B (B ⊆ query params) without rebuilding.
- A build-param override is rejected with a clear error.
- The evaluation's run manifest records build snapshot + overrides + effective config; the UI shows the effective config.
- Existing evaluations and indexes behave exactly as before.
