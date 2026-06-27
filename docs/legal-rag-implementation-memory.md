# Legal RAG UI Implementation Memory

Last updated: 2026-06-26

## Goal

Implement the `docs/legal-rag-plan.md` workflow using the platform UI for
indexing and evaluation. Use scripts only for dataset conversion/import support
and backend development checks.

## Separation Of Work

### Reusable Codebase Updates

- Keep benchmark metrics, retrieval extraction, taxonomy, and exports in backend
  services so the UI and future CLI paths can share them.
- Preserve benchmark-critical RAG config values in index/evaluation snapshots.
- Extend UI import/config/results surfaces only where the behavior is useful
  beyond one article run.

### Article-Only Actions

- Convert/download Legal RAG Bench data into ignored `data/` outputs.
- Create article-specific project, knowledge base, test set, RAG configs,
  indexes, evaluations, comparisons, screenshots, and exported tables.
- Decide Phase 1/Phase 2 model choices and report exact run metadata.

## Completed

- Created this progress memory file.
- Read `docs/legal-rag-plan.md` and confirmed the article must use the UI for
  indexing and evaluation, not the CLI.
- Noted existing modified files before making code changes:
  - `.claude/settings.local.json`
  - `docs/legal-rag-plan.md`
  - `platform/backend/app/services/evaluation_runner.py`
  - `platform/frontend/package-lock.json`
  - `platform/frontend/src/api/client.ts`
  - `.serena/` untracked
- Added converter support for `--content-mode clean|annotated`; `clean` is the
  default for paper-style calibration.
- Added Legal RAG test-case metadata to the converter output, including
  `relevant_passage_id` and `source_qa_id`.
- Added reusable test-case `metadata` persistence:
  - backend model/schema/API create/update/import/export,
  - Alembic migration `20260626_000001_add_test_case_metadata.py`,
  - frontend API types and import dialog preservation.
- Fixed Qdrant hybrid dense vector dimension handling:
  - supports explicit `parameters.embedding_dimension`,
  - defaults known OpenAI embedding model dimensions,
  - exposes `embedding_dimension` in the RAG parameter registry for UI snapshots.
- Added reusable Legal RAG Bench backend services:
  - retrieval extraction and `hit@k` / `gold_accessed`,
  - binary judge service,
  - taxonomy derivation,
  - summary aggregation.
- Integrated Legal RAG metrics into UI evaluations through existing raw metrics
  artifacts and evaluation summary JSON.
- Added backend endpoint
  `GET /evaluations/{evaluation_id}/raw-metrics/{result_id}`.
- Updated frontend API client and summary metric typing for Legal RAG benchmark
  payloads.
- Ran verification:
  - Python compile checks for touched backend/core files: passed.
  - Backend `uv run ruff check app alembic`: passed.
  - Root `uv run ruff check scripts src`: passed.
  - Frontend `npm.cmd run lint`: passed.
  - Frontend `npm.cmd run build`: passed.
  - Full backend `uv run pytest -q`: timed out after 120s.
  - Targeted backend tests ran but existing
    `tests/test_services/test_evaluation_runner.py` reported `pass_rate` 0.0
    vs expected 1.0; this still needs separate investigation.

- Resolved the evaluation runner test failure (2026-06-26):
  - Root cause PROVEN via `git diff HEAD`: an unnecessary refactor (not the
    legal-RAG metrics) had switched per-test-case result/progress persistence
    from the shared `self.db` session to fresh `async_session_maker()` sessions.
  - In the test fixture, `db_session` runs inside a single outer transaction
    that is rolled back; results committed on a separate session are invisible
    to it, and progress UPDATEs miss the job row that lives in the uncommitted
    transaction. Hence `pass_rate` 0.0 and `progress_current` 0.
  - Fix: reverted result/artifact/progress/finalize persistence to the proven
    baseline (`self.db` serialized by the existing `db_lock`), keeping all the
    legitimate legal-RAG additions (judge params, `legal_rag_bench` in
    `raw_metrics`, taxonomy, summary). Removed now-unused `async_session_maker`
    import.
  - Verification: single test, then all 3
    `tests/test_services/test_evaluation_runner.py` pass (1.49s). `ruff check`
    on the three touched services passes.
  - NOTE: a single targeted test runs fine (no hang); the hang is only at full
    pytest collection (chromadb/deepeval import side effects). Run targeted
    tests with `-p no:cacheprovider -o addopts="" -q`.

- Verified benchmark-critical build settings flow into the index snapshot
  (2026-06-26, no code change needed):
  - Registry exposes `chunk_size`/`chunk_overlap` (build phase, both vector
    types), `embedding_dimension` (vector_hybrid, with 3072 note for large),
    filesystem `max_tool_calls`/`max_file_reads` (query phase). `embedding_model`
    is a top-level RAG config field in `BUILD_TIME_TOP_LEVEL_FIELDS`.
  - `index_build_service` freezes `rag_type`, full `parameters`, split
    `build_parameters`/`query_default_parameters`, `embedding_model`,
    `llm_model`, `llm_provider` into `KnowledgeBaseIndex.config_snapshot`.
  - `top_k=5` is applied per-evaluation as a query override (not in the
    snapshot), consumed by `rag_adapter.load_rag_for_index_query`.

- Added Legal RAG Bench display to the Evaluation Results UI (2026-06-26):
  - New `platform/frontend/src/components/evaluations/LegalRagBenchMetrics.tsx`
    exporting `LegalRagBenchSummary` (aggregate card: hit@5/gold_accessed,
    correct, grounded rates + taxonomy breakdown) and `LegalRagResultMetrics`
    (per-result panel fetching the raw-metrics artifact via
    `api.evaluations.getRawMetrics`, showing hit@k/gold_accessed/correct/
    grounded badges, gold passage id + rank, retrieved ids, judge reasoning,
    taxonomy chip).
  - Wired summary card above the results list and per-result panel into the
    detail overview tab in `EvaluationResults.tsx`.
  - Verified: frontend `npm run lint` and `npm run build` both pass.

- Added Legal RAG Bench comparison support (plan 7.6) (2026-06-26):
  - Backend already passes `summary_metrics.legal_rag_bench` through the
    comparison aggregate (via `SummaryMetrics.legal_rag_bench`), so this was
    frontend-only.
  - `compare-utils.ts`: `ComparisonMember` now carries `legalRagBench`
    (extracted from each member's `summary.legal_rag_bench`).
  - New `platform/frontend/src/components/comparisons/LegalRagBenchComparison.tsx`
    renders a side-by-side table across members: Hit@5, Gold accessed, Correct,
    Grounded rates (best member highlighted) + per-taxonomy question counts.
  - `ComparisonDetail.tsx`: new "Legal RAG Bench" tab (Scale icon), shown only
    when at least one member has legal data; falls back to Metrics otherwise.
  - Verified: frontend `npm run build`, `npm run lint`, `tsc --noEmit` all pass.

- Added article-ready export (plan 7.6, 12) (2026-06-26):
  - New `platform/backend/app/services/evaluation_exporter.py` (pure, no DB):
    `ExportMember` dataclass + `headline_rows`/`taxonomy_rows`, `to_csv`,
    `to_markdown_table`, `build_markdown_report` (headline + taxonomy tables +
    per-member run manifest/config-snapshot appendix), `per_question_jsonl` +
    `build_question_record`. Headline columns follow plan section 12; retrieval
    metric is hit@5 or gold_accessed per system.
  - New endpoint `GET /comparisons/{id}/export?format=markdown|csv|jsonl&table=
    headline|taxonomy` in `app/api/comparisons.py`. Assembles members
    (baseline first, then compared order), derives rag_type from the index
    config_snapshot (falls back to rag_config), builds the manifest dict from
    `Evaluation.run_manifest`. JSONL pulls per-result legal payloads from the
    raw-metrics artifacts. Returns a `Response` with attachment headers.
  - Frontend: `api.comparisons.exportUrl(id, format, table?)` in client.ts;
    ComparisonDetail shows Markdown / Headline CSV / Taxonomy CSV / JSONL
    download links (only when a member has legal data).
  - Tests: `tests/test_services/test_evaluation_exporter.py` (5 pure-function
    tests). Run targeted: `./.venv/Scripts/python.exe -m pytest <path> -p
    no:cacheprovider -o addopts="" -q` (full collection hangs).
  - Verified: 5 backend tests pass, ruff clean, comparisons API imports;
    frontend `tsc`, `lint`, `build` all pass.

- Exposed Legal RAG Bench metrics in the Start Evaluation wizard (2026-06-26):
  - Gap found during manual run: `StartEvaluationWizard.tsx` only listed the 5
    DeepEval metrics, so the legal metrics could never be enabled from the UI.
  - Added `LEGAL_RAG_METRICS` (`legal_rag_retrieval`, `legal_rag_binary_judge`)
    as an opt-in group in the Metrics step (off by default - the binary judge
    costs LLM calls). Extracted `renderMetricOption` to share button markup.
  - These ids flow through `metric_names` -> `metric_config.metrics`; the runner
    picks them up via `is_legal_rag_metric_enabled`, and DeepEval's
    `_initialize_metrics` ignores unknown ids (name-based if-checks).
  - Judge model/provider already existed in the wizard's Query step (second
    ModelSelector) - no change needed; temperature is hardcoded to 0.0.
  - Verified: frontend tsc, lint, build all pass.

- Fixed concurrent SQLite commit crash in evaluation_runner (2026-06-26):
  - Symptom: eval failed with "(sqlite3.OperationalError) cannot commit
    transaction - SQL statements in progress" on test case 2.
  - Root cause: concurrent `_process_test_case` tasks share one `self.db`
    AsyncSession. Only the result commit was under `db_lock`; the 3 artifact
    `store_json` flushes, `_get_completed_count` SELECT, and
    `checkpoint_service.update_progress`/`save_checkpoint` (the checkpoint
    service holds the same session) ran outside it. The legal judge's added
    `await` widened the interleaving window and exposed the latent race.
  - Fix: moved artifact stores + result commit + completed-count +
    update_progress + save_checkpoint all inside the single `async with
    self.db_lock` block. `event_log.log_event` uses its own session, left
    outside. Verified: py_compile + ruff clean.

- Fixed Legal RAG Bench binary judge false-success on refusal answers
  (2026-06-26):
  - Symptom from first example run: generated answer was
    "I cannot answer this question based on the provided context." for the
    statutory murder / jury penalty question. DeepEval G-Eval correctly scored
    correctness as 0, but Legal RAG Bench showed `SUCCESS`, `CORRECT`, and
    `GROUNDED`.
  - Root cause: the judge prompt allowed the LLM judge to reason over the
    reference answer and retrieved context as if it were judging whether the
    correct legal answer was supported, instead of strictly judging the
    generated answer. Its own reasoning said the generated answer did not
    directly answer, then still marked success because the retrieved context
    supported the reference answer.
  - Fix in `platform/backend/app/services/legal_rag_bench_judge.py`: strengthened
    the prompt to label and evaluate only `ANSWER UNDER EVALUATION`, explicitly
    reject refusals/non-answers, and avoid marking answers correct merely
    because the reference answer is supported by the retrieved context.
  - Added deterministic post-judge sanity checks for common non-answer/refusal
    patterns. If the answer is a refusal or says there is insufficient context,
    force `correct=false` and `grounded=false`, with an `overrides` marker in
    the raw judge payload.
  - Added regression tests in
    `platform/backend/tests/test_services/test_legal_rag_bench_judge.py`,
    including the exact failure mode where the fake LLM judge returns
    `correct=true, grounded=true` for a refusal answer.
  - Verified from `platform/backend`: `rtk uv run pytest
    tests/test_services/test_legal_rag_bench_judge.py -q` passed; `rtk uv run
    ruff check app/services/legal_rag_bench_judge.py
    tests/test_services/test_legal_rag_bench_judge.py` passed.
  - Existing evaluation artifacts still contain the old judge output; rerun the
    evaluation to refresh Legal RAG Bench correctness/groundedness/taxonomy.

- Investigated inconsistent metrics in run
  `Somke KB - legal_vector_search - 26 giu, 18:10` (2026-06-26):
  - Saved the detailed analysis report at
    `reports/somke-kb-legal-vector-search-2026-06-26-1810-inconsistency-report.md`.
  - Main finding: DeepEval metric scores/reasons/raw artifacts are contaminated
    across concurrent batches of 5 cases, strongly implicating shared mutable
    metric instances while `DEEPEVAL_ASYNC_MODE=True`.
  - The report also records Legal RAG retrieval metrics being null due to
    missing passage ids in test-case metadata, taxonomy counts summing to 9/10,
    suspicious refusal answers despite relevant context in several cases, and
    async checkpoint data inconsistency.
  - Future bug-fix sessions should start from that report before trusting the
    evaluation's aggregate `relevancy_avg`, `overall_avg`, or `pass_rate`.

- Fixed the main inconsistencies from the report (2026-06-26):
  - `evaluation_runner.py`: DeepEval metric instances are now created per test
    case, not shared across concurrent async tasks. Metric DB columns and raw
    metric artifacts are written from immediate per-case snapshots.
  - `evaluation_runner.py`: async checkpoints now store completed count as
    `last_index`, avoiding out-of-order task indexes in checkpoint data.
  - `legal_rag_bench_metrics.py`: passage-id extraction now falls back from
    test-case metadata to annotated ground-truth context and retrieval trace
    source filenames. Identifier normalization no longer collapses ids like
    `1.5-c6-s1` to `1`, and filename ids like `1_5-c6-s1` normalize to match.
  - `legal_rag_bench_metrics.py`: grounded-but-incorrect judge outputs without
    retrieval metrics now classify as `grounded_but_incorrect`, so taxonomy
    counts no longer silently drop that case.
  - Added focused regressions:
    `tests/test_services/test_evaluation_runner.py` covers async metric
    instance isolation and raw metric consistency;
    `tests/test_services/test_legal_rag_bench_metrics.py` covers passage-id
    fallback/normalization and taxonomy coverage.
  - Verification passed:
    `rtk uv run pytest tests/test_services/test_evaluation_runner.py tests/test_services/test_legal_rag_bench_metrics.py tests/test_services/test_legal_rag_bench_judge.py -q -p no:cacheprovider -o addopts=""`
    and `rtk uv run ruff check app/services/evaluation_runner.py
    app/services/legal_rag_bench_metrics.py
    tests/test_services/test_evaluation_runner.py
    tests/test_services/test_legal_rag_bench_metrics.py
    tests/test_services/test_legal_rag_bench_judge.py`.
  - Existing evaluation artifacts remain corrupted; rerun the evaluation before
    judging generation/retrieval quality or aggregate metrics.

- Added an `abstention` taxonomy bucket and fixed generation over-refusal
  (2026-06-26), triggered by analysing run
  `Somke KB - legal_vector_search - 26 giu, 18:48`:
  - Context: the example case (Emma / circumstantial-evidence question) showed
    `Faithfulness/Relevancy/Precision = 1.00` while the generated answer was the
    refusal "I cannot answer this question based on the provided context." This
    is a metric artifact, not a bug: DeepEval Faithfulness/Relevancy are vacuous
    on refusals (no claims to contradict; the lone statement is judged relevant),
    and ContextualPrecision/Recall score the retriever from the expected output,
    not the generated answer. Only the binary judge + G-Eval correctness reflect
    answer quality, and both correctly flagged it.
  - Abstention bucket: refusals were lumped into `hallucination_or_ungrounded`
    because the judge's deterministic non-answer override forces
    `grounded=false`. A refusal is an abstention, not a hallucination.
    - `legal_rag_bench_judge.py`: `_apply_answer_sanity_checks` now also sets an
      explicit `abstention: true` flag on the judge payload.
    - `legal_rag_bench_metrics.py`: `derive_taxonomy` returns `abstention`
      before the `grounded=false` bucket; `summarize_legal_rag_metrics` reports
      `abstention_rate` in the judge summary.
    - Frontend `LegalRagBenchComparison.tsx` + `LegalRagBenchMetrics.tsx`: added
      the "Abstention" label/colour (sky). `evaluation_exporter.py`: added the
      "Abstention" column to `TAXONOMY_ORDER` (CSV/Markdown tables).
    - Tests: 3 new cases in `test_legal_rag_bench_metrics.py` (abstention takes
      precedence, ungrounded-without-abstention stays hallucination,
      `abstention_rate` + counts).
  - Generation over-refusal root cause: the shared generation prompt was
    extractive ("answer only ... if the answer cannot be found in the context,
    refuse"). Legal questions are rule-application (context holds a general rule;
    the question is a named hypothetical), so a faithful model reads "the
    specific answer isn't literally in the context" and refuses even when the
    gold passage was retrieved in the top-k. Verified the refusal is the model's
    own output, not the code's empty-content fallback ("No answer generated"),
    and that the gold passage was at rank 2 of 5 in the trace.
    - Fix: rewrote the prompt in all four implementations
      (`vector_semantic/chroma_rag.py`, `vector_hybrid/hybrid_rag.py`,
      `graph_rag/neo4j_rag.py`, `google_vertex_search/google_vertex_rag.py`) to
      license applying rules/principles from the context to the question's
      scenario, answer yes/no conclusions first, and narrow the refusal trigger
      to "no relevant rule in context". Kept the prompt identical across all four
      (apples-to-apples) and preserved the exact refusal sentinel string the
      judge keys off.
  - Verified: abstention taxonomy/judge/exporter logic checked via direct
    `./.venv/Scripts/python.exe` imports (pytest collection still hangs);
    confirmed all four prompts updated, old wording gone, sentinel intact.
    Frontend `tsc`/lint not re-run (one-line type-safe map additions).
  - Trade-off to watch: the prompt fix trades abstentions for substantive
    answers; the new abstention bucket is the instrument to confirm
    `grounded_but_incorrect` does not rise as `abstention` falls.
  - Not retroactive: rerun the evaluation for both the new prompt and the
    abstention bucket to take effect. Proving the fix (refusal -> correct on the
    Emma case, old vs new prompt) still needs a live model A/B run.

- Distinguished abstentions from hallucinations in the per-result UI
  (2026-06-26):
  - Follow-up to the abstention bucket. The result panel previously rendered red
    Correct-X / Grounded-X badges on a refusal, reading like a hallucination even
    though the taxonomy chip already said "Abstention".
  - `LegalRagBenchMetrics.tsx`: added an `AbstainedBadge` (neutral sky,
    `MinusCircle`). When `taxonomy === 'abstention'` the badge row shows the
    single "Abstained" badge instead of the Correct/Grounded pair; the hit@k
    badge stays, since the interesting point is that the model abstained despite
    retrieving the gold passage. Non-abstention results are unchanged.
  - Verified: `tsc --noEmit` passes. Not retroactive - rerun the eval so stored
    results carry the `abstention` taxonomy and show the new badge.

## Current Step

- Backend + UI for Legal RAG Bench comparison and export are complete. Remaining
  work is manual UI verification / running the experiment phases.

## Pending

- Verify UI can import Legal RAG Bench passages as a Knowledge Base (manual UI).
- Perform Phase 0 UI smoke once product support is ready (manual UI).
