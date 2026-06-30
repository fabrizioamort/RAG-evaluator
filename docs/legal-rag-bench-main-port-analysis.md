# Legal RAG Bench — Work Analysis & Main-Branch Port Plan

**Date:** 2026-06-27
**Branch analyzed:** `legal-rag-bench` (commit `3b80126`, branched from `main` @ `5ee089a`)
**Scope:** 34 files, +3473 / -205 lines

## Purpose

All Legal RAG Bench work lives on the `legal-rag-bench` branch and exists to
produce the comparison **article** — it is not needed in the product itself.
However, while building the benchmark we also fixed real bugs and made general
improvements that **do** belong in `main`.

This report separates the two so the benchmark stays on its own branch while the
genuine fixes get cherry-picked into `main`.

## TL;DR — what to port to `main`

| # | Change | Files | Why it belongs in main |
|---|--------|-------|------------------------|
| 1 | **Per-test-case metric isolation** | `evaluation_runner.py`, `test_evaluation_runner.py` | Concurrency correctness bug: DeepEval metric objects are mutable and were shared across concurrent cases. |
| 2 | **Full db-lock coverage in async runner** | `evaluation_runner.py` | Concurrency bug: SQLite "cannot commit transaction" when concurrent tasks share one session. |
| 3 | **Checkpoint `last_index` correctness** | `evaluation_runner.py` | Resume bug under concurrency (uses real completed count, not the task index). |
| 4 | **Hybrid RAG embedding dimension** | `hybrid_rag.py`, `registry.py` | Real bug: Qdrant dense size hardcoded to 1536, breaks `text-embedding-3-large` (3072). |
| 5 | **Batched multipart upload** | `client.ts` | Real bug: Starlette rejects >1000 file parts; large folder uploads fail. |
| 6 | **Anti-over-refusal answer prompt (all 4 RAGs)** | `chroma_rag.py`, `hybrid_rag.py`, `neo4j_rag.py`, `google_vertex_rag.py` | Reduces spurious refusals when retrieval is good. *(Recommended — verify on non-legal sets first.)* |
| 7 | **Generic `metadata` column on test cases** | `test_case.py`, migration, `schemas/test_set.py`, `api/test_sets.py`, `ImportTestSetDialog.tsx`, `client.ts` | Reusable, self-contained feature; "legal" only by current usage. *(Optional — needs the Alembic migration.)* |

Everything else is **article-only** and should stay on this branch.

---

## Category A — General fixes & improvements → PORT to `main`

### 1. Per-test-case metric isolation (concurrency bug fix) — HIGH PRIORITY
**File:** `platform/backend/app/services/evaluation_runner.py`

Previously a single `metrics` list was built once in `run()` and reused by every
concurrent `_process_test_case` call. DeepEval metric objects are **mutable** —
`a_measure()` writes `.score` / `.reason` onto the instance. Under
`DEEPEVAL_ASYNC_MODE` with concurrency > 1, two cases racing on the same metric
object overwrite each other's scores/reasons → **wrong, cross-contaminated
results**.

Fix: `_initialize_metrics()` is now called **fresh inside each
`_process_test_case`**, so every case gets isolated metric instances. The judge
model/provider/base-url/api-key are threaded into `_process_test_case` because
metric construction moved there.

**Regression test:** `test_async_runner_uses_isolated_metric_instances` runs two
cases concurrently with different scores/latencies and asserts no leakage. This
test is general (no legal dependency) and should be ported too.

### 2. Full db-lock coverage in the async runner (concurrency bug fix) — HIGH PRIORITY
**File:** `platform/backend/app/services/evaluation_runner.py`

Concurrent tasks share one `AsyncSession` (`self.db`). Previously the artifact
`store_json(...)` calls (which flush) ran **outside** `self._db_lock`, while only
the final commit was inside it. One task flushing mid-statement while another
committed produced SQLite `cannot commit transaction - SQL statements in
progress`.

Fix: **all** db-touching statements — the three artifact stores, the result
commit, the progress read, and the checkpoint write — are now inside
`self._db_lock`.

### 3. Checkpoint `last_index` correctness — port with #2
**File:** `platform/backend/app/services/evaluation_runner.py`

Checkpoint changed from `{"last_index": i + 1}` (the task's own index — meaningless
for resume under out-of-order concurrent completion) to the actual completed
count, written under the lock.

> **Porting note for #1–#3:** these hunks live in the same file as the
> article-only legal logic. Cherry-pick the hunks, not the whole file. **Keep:**
> per-case `_initialize_metrics`, judge-param threading, the `_db_lock`
> widening, the checkpoint change, and the `_metric_result_field_name` helper.
> **Drop:** the `legal_rag_judge` import/init, the `legal_rag_result` block, the
> `raw_metrics["legal_rag_bench"]` injection, `_collect_legal_rag_summary`, and
> the legal summary in `finalize`.

### 4. Hybrid RAG embedding dimension — real bug fix
**Files:** `src/rag_evaluator/rag_implementations/vector_hybrid/hybrid_rag.py`,
`src/rag_evaluator/rag_implementations/registry.py`

Qdrant's dense vector size was hardcoded `size=1536` ("text-embedding-3-small
dimension"). Any larger model (`text-embedding-3-large` = 3072) silently
mismatched and broke indexing. Fix adds:
- `EMBEDDING_DIMENSIONS` model→dim map + `_resolve_embedding_dimension()` (reads
  `parameters.embedding_dimension`, else the map, else raises clearly).
- A new `embedding_dimension` **build** parameter in `registry.py` (default 1536,
  documents 3072 for the large model).

Self-contained and clearly correct. Port as-is.

### 5. Batched multipart document upload — real bug fix
**File:** `platform/frontend/src/api/client.ts` (`knowledgeBases.uploadDocuments`)

Starlette rejects multipart requests with **>1000 file parts**, so uploading a
large folder failed outright. Fix uploads in batches of 500 and aggregates the
responses. General fix for any large knowledge base. Port as-is.

### 6. Anti-over-refusal answer-synthesis prompt — RECOMMENDED (verify first)
**Files:** `chroma_rag.py`, `hybrid_rag.py`, `neo4j_rag.py`,
`google_vertex_rag.py` (identical change, graph wording in neo4j)

The strict extractive prompt ("If the answer cannot be found in the context,
say...") caused refusals even when retrieval succeeded. Replaced in all four RAGs
with: apply general rules/principles in the context to the specific situation;
for yes/no questions give the conclusion first, then the supporting rule; refuse
**only** when the context has nothing relevant.

This is a genuine quality fix (see memory `legal-rag-overrefusal-and-abstention`),
but the wording leans toward reasoning/QA. **Recommendation:** port to `main`,
but first sanity-check it does not loosen grounding on a strict factual test set.

---

## Category B — Borderline generic infrastructure → OPTIONAL port

### 7. Generic `metadata` column on test cases
**Files:** `platform/backend/app/models/test_case.py`,
`platform/backend/alembic/versions/20260626_000001_add_test_case_metadata.py`,
`platform/backend/app/schemas/test_set.py`,
`platform/backend/app/api/test_sets.py`,
`platform/frontend/src/components/test-sets/ImportTestSetDialog.tsx`,
`platform/frontend/src/api/client.ts` (`TestCase` / `TestCaseCreate`)

Adds a generic `metadata` JSON column (mapped as `metadata_` to avoid the
SQLAlchemy reserved name) plumbed through create / bulk / update / import /
export. It was added to carry Legal RAG Bench fields (e.g. `relevant_passage_id`,
`source_qa_id`), but the design is generic and well isolated.

**Recommendation:** optional but reasonable to port — it is harmless and reusable.
If ported, the **Alembic migration must go to `main`** and the frontend `metadata`
interface fields come along.

### `getSummaryScore` / `renderMetricOption` refactors — optional, port only if you port the field they support
- `EvaluationResults.tsx` adds a typed `getSummaryScore` accessor that tolerates
  the now-non-numeric `legal_rag_bench` entry on `SummaryMetrics`. Only needed if
  the `legal_rag_bench` schema field is ported (it should **not** be).
- `StartEvaluationWizard.tsx` extracts `renderMetricOption` to add the legal
  metrics group. Clean refactor on its own, but introduced for the legal group;
  low value to port in isolation.

### `GET /evaluations/{id}/raw-metrics/{result_id}` — optional
**File:** `platform/backend/app/api/evaluations.py`

Returns the existing `raw_metrics` artifact for a result. Added to surface legal
payloads to the UI, but the endpoint itself is generic and could serve as a
debugging aid in `main`. Low priority.

---

## Category C — Legal RAG Bench ONLY → KEEP on branch, do NOT port

### Backend (new files)
- `app/services/legal_rag_bench_judge.py` — binary LLM judge.
- `app/services/legal_rag_bench_metrics.py` — hit@k / retrieval / taxonomy /
  `summarize_legal_rag_metrics`.
- `app/services/evaluation_exporter.py` — article export (Markdown / CSV / JSONL).
- `tests/test_services/test_legal_rag_bench_judge.py`
- `tests/test_services/test_legal_rag_bench_metrics.py`
- `tests/test_services/test_evaluation_exporter.py`

### Backend (modifications — legal hunks only)
- `app/api/comparisons.py` — `/comparisons/{id}/export` endpoint + export-member
  helpers (`_to_export_member`, `_build_question_records`, etc.).
- `app/api/evaluations.py` — `/raw-metrics/{result_id}` (see Category B note).
- `app/services/evaluation_runner.py` — legal judge init, `legal_rag_result`
  block, `raw_metrics["legal_rag_bench"]`, `_collect_legal_rag_summary`, legal
  summary in `finalize`.
- `app/schemas/evaluation.py` — `SummaryMetrics.legal_rag_bench` field.

### Frontend (new files)
- `components/comparisons/LegalRagBenchComparison.tsx`
- `components/evaluations/LegalRagBenchMetrics.tsx`

### Frontend (modifications — legal hunks only)
- `components/comparisons/ComparisonDetail.tsx` — legal section + export buttons.
- `components/comparisons/compare-utils.ts` — `legalRagBench` aggregation.
- `components/evaluations/EvaluationResults.tsx` — legal summary + per-result
  legal metrics rendering.
- `api/client.ts` — `getRawMetrics`, export URL helper, `legal_rag_bench` fields.

### Scripts & docs
- `scripts/convert_legal_rag_bench.py` — `clean`/`annotated` content modes +
  metadata emission (benchmark converter).
- `docs/legal-rag-plan.md`, `docs/legal-rag-implementation-memory.md`,
  `docs/legal-rag-ui-walkthrough.md`.

---

## Recommended porting workflow

1. Branch `port/runner-and-rag-fixes` off `main`.
2. **Pure adds (safe, no entanglement):** items 4 and 5 — `hybrid_rag.py` +
   `registry.py` embedding dimension, and `client.ts` batched upload.
3. **Runner hunks:** apply items 1–3 by hand (cherry-pick hunks; drop every legal
   block per the porting note above) and bring the
   `test_async_runner_uses_isolated_metric_instances` test.
4. **Prompt (item 6):** apply to all four RAGs; run an existing strict factual
   test set to confirm no grounding regression.
5. **Optional (item 7):** the generic `metadata` column + its migration, if
   wanted.
6. Verify per CLAUDE.md (backend `uv run pytest` from `platform/backend`;
   `uv run ruff check .`; frontend `npm run lint`). Note: `uv run pytest` at repo
   root hangs on this machine — run backend tests from `platform/backend`.

## Excluded from the commit
- `.serena/` (Serena MCP tool config) — left untracked, not part of the work.
- Pre-existing stash `codex-backup-before-pull-main` — untouched.
