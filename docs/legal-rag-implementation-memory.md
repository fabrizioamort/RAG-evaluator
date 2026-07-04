# Legal RAG UI Implementation Memory

Last updated: 2026-07-04

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

- Reviewed and optimized Filesystem RAG retrieval on
  `Somke KB - legal fs - 27 giu, 11:06` (2026-06-27):
  - Evaluation id: `cb69162e74e1430ca59bc5cf6ca84d1b`; index:
    `Somke KB - legal fs` / physical id `idx_bc09e1932b5a4fc5bd316e62`;
    query overrides used `openai/gpt-5.4-mini` via OpenRouter with
    `max_iterations=20`, `max_tool_calls=30`, `max_file_reads=30`.
  - Aggregate metrics were strong (`overall_avg=0.895`) but `pass_rate=0.5`;
    manual result inspection showed most generated answers were usable while
    retrieved context was often summary-heavy.
  - Root cause for the Sally backyard failure: the correct answer is `view`.
    The source says a court may order a "demonstration, experiment or
    inspection" collectively called a "view"; `inspection` is the narrower
    subtype for travelling to a location/object. The FS trace had found
    `doc_045` (`2.1 Views`) but only injected `_summaries/doc_045_summary.md`,
    so the model latched onto the subtype and answered `Inspection`.
  - `filesystem_rag/agent/agent.py`: deterministic lexical prefetch now weights
    high-signal exact-name terms (`view`, `views`, `inspection`,
    `demonstration`, `experiment`, `procedure`, plus existing legal/VARE
    signals), boosts title matches, handles hyphenated terms such as
    `self-incrimination`, and injects focused full-document excerpts for the top
    two candidate docs in addition to summaries.
  - `filesystem_rag/agent/prompts.py`: summaries are now framed as navigation
    aids. For exact-name/legal-procedure questions, the agent should verify
    against full text/excerpts and prefer named collective statutory/legal terms
    over narrower examples.
  - `tests/unit/test_filesystem_rag_agent.py`: added a regression fixture for
    the Sally question. It asserts `doc_045` ranks first and
    `documents/doc_045.md` is included with the phrase
    `collectively, a "view"`.
  - Verification passed:
    `rtk uv run pytest tests/unit/test_filesystem_rag_agent.py -q`;
    `rtk uv run ruff check src/rag_evaluator/rag_implementations/filesystem_rag/agent/agent.py src/rag_evaluator/rag_implementations/filesystem_rag/agent/prompts.py tests/unit/test_filesystem_rag_agent.py`;
    and a live deterministic prefetch check against the prepared
    `idx_bc09e1932b5a4fc5bd316e62` index now returns `doc_045` first and includes
    both `_summaries/doc_045_summary.md` and `documents/doc_045.md`.
  - Scope note: the full-document excerpt change is general and should help any
    corpus where summaries are lossy or exact wording matters. The current term
    weights are partly legal-benchmark tuned; a more general future improvement
    is corpus-adaptive lexical scoring or a proper local BM25/full-text index.
  - Indexing-phase follow-up identified: `question_seeds.md` still contains
    low-value generic seeds (for example "What is not?"), topic labels are
    generic (`science`, `business`, `technical`) for legal content, summary
    section ranges can be noisy (`lines X-0`), entity extraction is noisy, and
    Legal RAG passage ids/titles should be stronger first-class metadata.

- Fixed Legal RAG Bench `hit@k` undercount (metric bug, not retrieval) and
  backfilled the affected clean run (2026-06-28):
  - Symptom: the full clean run (`vector_semantic`, text-embedding-3-large,
    chunk 8000/0, deepseek-v4-flash) reported Hit@5 38%, well below the paper's
    52%. Re-indexing in clean content mode had not helped (annotated was 41%).
  - PROVEN it was a measurement bug, not retrieval quality. Replicated the
    paper's pipeline offline against the vectors already stored in Chroma
    (`platform/backend/storage/indexes/<physical_id>/chroma`, collection name =
    physical_id). Self-check re-embedding a stored passage gave cosine 1.0000
    (same embedding space; note embeddings route through OpenRouter because the
    index config `embedding_base_url=None` falls back to core
    `settings.openai_base_url`). Both exact brute-force cosine top-5 AND Chroma's
    native HNSW query returned hit@5 = 52% = paper exactly. So embeddings, data,
    chunking (1:1, 4876), content mode, distance (cosine), dims (native 3072),
    and gold-id matching (0/100 failures) are all correct; content mode barely
    mattered. Both pipelines are bare dense retrieval - no query transformation
    or reranking on either side.
  - Root cause in `legal_rag_bench_metrics.py`: `extract_retrieved_passage_ids`
    /`_extract_ids_from_value` emitted MULTIPLE ids per retrieved chunk - the
    real passage id (from `source`) PLUS a synthetic `doc_<hash>` (from
    `metadata.doc_key`) PLUS the raw context text - so for top_k=5 the 5 real
    passages landed at list-ranks 1,3,5,7,9. `hit_at_k` (rank<=top_k) then could
    not see the 4th/5th passage; `gold_accessed` (membership) was immune. The
    stored summary already showed it: `hit_at_k_rate=0.38` vs
    `gold_accessed_rate=0.52`. `base_rag.query_with_trace` returns `metadata`
    with NO `sources` key, so the only ranked id source is
    `retrieval_trace.retrieved_chunks`.
  - Fix (backend only, no RAG/core/re-index change):
    - `_extract_ids_from_value` now emits EXACTLY ONE id per retrieved item,
      preferring an explicit passage id, then the document source path, then a
      synthetic id, and recurses container keys (`retrieved_chunks`, etc.) in
      rank order.
    - `extract_retrieved_passage_ids` now only mines `response["context"]` text
      for ids as a fallback when no structured source was found (prevents large
      text blobs polluting the ranked list).
    - Regression test `test_retrieval_metrics_rank_uses_one_id_per_chunk` in
      `tests/test_services/test_legal_rag_bench_metrics.py` (the prior trace test
      used a single chunk, where the real id is always rank 1, so the multi-chunk
      interleaving slipped through).
  - Verified: replaying the patched code over the real stored traces moved
    hit@k 38% -> 52% (= `gold_accessed`); existing metric tests + the new one
    pass via direct `./.venv/Scripts/python.exe` import (pytest collection still
    hangs).
  - Backfilled the clean run `06c2f8da7a9647e384dec0883c6edc61` (index physical
    id `idx_d4f8194a3f2643229ac9eaac`) in one transaction (dev.db backed up
    first): recomputed retrieval + taxonomy per question from the stored traces
    (judge untouched), wrote new `raw_metrics` artifacts and repointed all
    100/100 result rows, and rewrote `evaluation.summary_metrics`. Result:
    Hit@5 38% -> 52%; taxonomy retrieval_error 10 -> 8, reasoning_error 6 -> 8
    (2 grounded-but-incorrect cases with gold at rank 4/5 reclassify);
    abstention 18 / hallucination_or_ungrounded 17 / success 49 unchanged.
    correct 49% / grounded 65% unchanged (judge not re-run).
  - Takeaway: the RAG reproduces the paper's retrieval. The remaining distance to
    the paper is generation/judge model strength (deepseek-v4-flash vs GPT-5.2)
    plus the 18 abstentions - not retrieval.
  - Not yet done: the same bug inflated the annotated run (`eee46010...`) and any
    earlier legal evals (only the clean run was backfilled); the metric fix
    should be ported to `main` per the branch/port plan.

- Diagnosed the correctness gap vs the paper as a closed-book vs open-book
  generation difference (2026-06-28):
  - Trigger: deepseek-v4-pro scored Correct 60% on the full 100 and on the
    first-10 subset; gemini-3.1-pro scored Correct 60% on the same 10 subset -
    a frontier model tied with a weaker one, and well below the paper's
    gemini/GPT-5.2 correctness (76-77%) for text-embedding-3-large.
  - METHOD CAVEAT found first: the user used the SAME model for generation AND
    judging (deepseek judged deepseek, gemini judged gemini). The paper holds the
    judge FIXED (GPT-5.2 high-reasoning) for every generator/embedder. Self-judge
    makes runs non-comparable to each other and to the paper. Rule for the
    article: always use ONE fixed strong judge (GPT-5.2) for every run; the
    wizard's Query step already has a separate judge ModelSelector.
  - Free inspection of stored results (eval ids: gemini-10
    `5ef2555e20e4491b8b98cbcb72b1f7b0`, deepseek-10
    `de77b5975a174fec9b61247c69205c35`, deepseek-100
    `409e6faa47ac4d4d93cb2d0b5a237c8a`): on the 10 subset both models fail the
    SAME 4 questions, and all 4 are gold-NOT-retrieved -> both ABSTAIN ("I cannot
    answer this question based on the provided context.") -> judge correctly marks
    them wrong. The 6 gold-retrieved questions are correct for both. So on this
    subset Correct % == retrieval % (6/10) for both a weak and a frontier model.
  - Verified the 4 misses are TRUE retrieval misses: the retrieved top-5 holds
    topically-adjacent but non-answering passages (e.g. the "view" question
    `2.1-c1-s1` retrieved cultivation-of-narcotics element passages, none
    defining a view; VARE/penalty/expert questions similar). The abstentions are
    correct given a context-only policy - the answer genuinely is not in context.
  - CONCLUSION: judge is fine and generator strength is not the differentiator.
    The gap to the paper is the generation policy. Our shared prompt is
    closed-book ("apply rules from the context; if the context has no relevant
    rule, reply exactly 'I cannot answer...'"), which CAPS correctness near the
    retrieval rate (plus redundant-passage overlap). The paper's gemini reaches
    77% with the SAME 52% retrieval by answering retrieval-misses from PARAMETRIC
    knowledge (it simply knows "a view", "VARE", "do not disclose the penalty");
    for these 4 there is no covering passage, so the recovery must be the model's
    own knowledge, not redundant context. So the paper is effectively open-book.
  - Cost implication: re-judging these stored answers with GPT-5.2 will NOT change
    them (genuine "I cannot answer" is wrong under any judge), and running gemini
    on the full 100 with the current closed-book prompt will NOT reach 77% (it
    will abstain on every retrieval miss and land near the retrieval rate). The
    lever is the generation policy, not the model or the judge.
  - DESIGN DECISION for the article (open):
    - Closed-book (current): Correct ~= retrieval + overlap. Stricter, cleaner
      test of the RAG pipeline; does not let pretrained legal knowledge mask
      retrieval gaps. Best for fairly comparing the 3 RAG architectures.
    - Open-book / paper-style: allow parametric knowledge ("use the context if
      relevant, otherwise answer from your own legal knowledge"); Correct can far
      exceed retrieval but partly measures the LLM's memorised law, not the RAG.
      Required to reproduce the paper's correctness; pair with a fixed GPT-5.2
      judge for every run.
  - Cheap confirmatory test not yet run: regenerate ONLY the 4 abstained
    questions with gemini under an open-book prompt (~4-8 calls); if it answers
    "view"/"VARE"/"No - do not disclose penalty" correctly, that proves
    closed-vs-open-book is the entire divergence. Relates to the anti-over-refusal
    prompt work above (that fix reduced refusals when the gold WAS retrieved; it
    did not - and by design should not - make the model answer when the context
    lacks the rule).

- Fixed Filesystem RAG `gold_accessed` always being False (2026-06-28):
  - Symptom: in a Filesystem RAG legal eval the gold-accessed check never fired -
    the gold passage id is a citation like `1.5-c8-s1`, but the retrieved ids the
    UI showed were synthetic doc ids (`doc_182`, `doc_182_summary`). Different
    namespaces, so `_normalize_identifier` in `legal_rag_bench_metrics.py` could
    never match and `gold_accessed` stayed False for every question.
  - Root cause (PROVEN against real data, no guessing): Filesystem RAG regroups
    each passage into `documents/doc_NNN.md` during preparation, so the agent
    reports `doc_NNN`/`doc_NNN_summary` sources. The passage id is not in those
    ids. The metric's existing normalization was already correct for the OTHER
    RAGs only because they index the raw `passage_NNNN__<id>.txt` files, whose
    source paths carry the `__<id>` suffix that `_identifier_from_path_like_value`
    strips to the gold id.
  - The doc->passage mapping already existed on disk: each
    `documents/doc_NNN.meta.json` has `"original_file"` pointing at the source
    passage file (e.g. `..._passage_0023__1_4-c1-s1.txt`), whose name embeds the
    passage id after `__`. The metric already strips the `__` prefix and converts
    `_`->`.` between digits, so it matches once the source is resolved.
  - Fix (RAG-side only, `filesystem_rag/filesystem_rag.py`; no metric change):
    added `_resolve_source_to_passage`/`_resolve_sources` (cached via
    `self._passage_source_cache`) that maps each `doc_NNN`/`doc_NNN_summary`
    source to its `original_file` by reading the doc's `meta.json`. Wired into
    `_context_from_agent_response` (chunk sources + retrieval trace) and the
    `context_sources`/`files_read` metadata in both `query` and
    `query_with_trace`. Non-document sources (index files) and a missing/bad
    `meta.json` pass through unchanged (try/except), so it degrades safely.
  - Verified end-to-end against the real index `idx_a1961ec0bf594fc1938c46a4`
    (`doc_034` -> `1.4-c1-s1`): `gold_accessed` flips False -> True, retrieved id
    shows `1_4-c1-s1` (matches gold `1.4-c1-s1` after the metric's `_`->`.`
    normalization). py_compile passes; the real class method resolves
    doc/summary/bare ids and passes index files through; the existing
    `test_query_with_trace_uses_single_agent_call` is unaffected (its fixture has
    no `doc_029.meta.json`, so the resolver returns the source unchanged). Pytest
    not run (collection still hangs); verified via direct
    `./.venv/Scripts/python.exe` imports.
  - `hit_at_k` stays None for Filesystem RAG by design - agentic file reads are
    not a ranked top-k list, so only `gold_accessed` is meaningful. Not
    retroactive: rerun the FS eval for stored results to carry the resolved ids.

- Added retry UI for partial failed evaluations (2026-06-28):
  - Trigger: evaluation `4eaffc78-564e-483f-8c50-9d4247b4b742` failed after
    saving 9/10 test cases and the backend error correctly said retry would run
    only the missing case, but the UI exposed no retry action.
  - `EvaluationProgress.tsx`: failed/cancelled evaluations now show a `Retry`
    button that calls `api.evaluations.retry`, reconnects the SSE stream, and
    keeps showing the saved-result count/error message when the page is
    reopened after a refresh.
  - `EvaluationDetail.tsx` and `ProjectDetail.tsx`: pass the fetched evaluation
    snapshot into the progress panel and invalidate evaluation queries after a
    retry starts.
  - `useEvaluationStream.ts` and frontend API types: handle `test_case_error`
    and `cancelled` events explicitly, clear stale errors when a new run starts,
    and reset local stream state on reconnect.
  - Backend retry now clears the in-memory job-event replay cache before
    starting the background retry, preventing an old failure SSE event from
    immediately closing the new retry stream.
  - Verification passed:
    `rtk npm run build`, `rtk npm run lint`,
    `rtk uv run pytest tests/test_api/test_evaluations.py::TestEvaluationControl::test_retry_incomplete_completed_evaluation -q`,
    and `rtk uv run ruff check app/api/evaluations.py app/services/job_event_log.py`.

- Diagnosed and fixed missing/hidden Legal RAG judge results in
  `Legal RAG clean - filesystem - 28 giu, 22:35` (2026-07-02):
  - Evaluation id: `4d5e5ad0-66bb-461d-8a4d-f65f548b5058`.
  - Initial user-visible symptom: the UI appeared to show only 88 Legal RAG
    classified results out of 100, no failed tests, and no retry option.
  - DB check showed the evaluation itself was complete: 100
    `evaluation_results`, 100 test cases, job status `completed`, and
    `summary_metrics.legal_rag_bench.judge.count=100`.
  - The apparent "88" came from taxonomy buckets summing only classified rows:
    12 judge outputs had `correct=None`, `grounded=None`, and
    `parse_error="judge_response_not_json"` with no taxonomy. One additional
    raw parse error had been deterministically classified as `abstention`, so
    there were 13 raw judge parse failures total.
  - Retry was unavailable by design: the evaluation was complete and had all
    100 result rows, so the normal evaluation retry endpoint correctly saw no
    missing test cases.
  - Fixed stale success/error state: `job_checkpoint_service.complete_job()` now
    clears `Evaluation.error_message` and `EvaluationJob.error_message` on
    successful completion.
  - Fixed silent judge drops:
    - `legal_rag_bench_metrics.py`: judge parse errors and null
      `correct`/`grounded` now map to taxonomy `judge_error`.
    - Legal summary now reports `judge.scored_count`,
      `judge.parse_error_count`, `classified_count`, and
      `unclassified_count` when relevant.
    - Exporter/UI/comparison components now display `judge_error` explicitly.
  - Improved judge robustness:
    - `LegalRAGBenchJudge.judge()` retries unparsable/empty judge responses up
      to 3 times and accumulates token/cost usage across attempts.
    - Added optional per-call timeout support for repair scripts.
  - Added repair script
    `platform/backend/scripts/repair_legal_rag_judge_errors.py`:
    reuses stored question, expected answer, generated answer, and retrieved
    context artifacts to rerun only the Legal RAG binary judge for failed rows.
    On per-row provider exception it stores a transparent `judge_exception`
    payload classified as `judge_error` and continues.
  - Verification passed:
    backend py_compile, ruff, targeted judge/metrics tests; frontend lint/build.

- Diagnosed and fixed oversized Filesystem RAG context leaking into judge calls
  (2026-07-02):
  - User provided an OpenRouter prompt example showing requests with more than
    1M input tokens.
  - Root cause: Filesystem RAG could read the full
    `_index/questions/question_seeds.md` file. In the affected index
    (`idx_844272ee548a4fe19c05ad52`) that file was 3,299,374 bytes; the saved
    OpenRouter prompt was 3,462,116 bytes.
  - Leak path:
    - Agent prompt encouraged checking `_index/questions/question_seeds.md`.
    - `read_file()` returned full file content when no range/header option was
      supplied.
    - `agent.py` stored every `read_file` result into `context_chunks`
      unchanged.
    - `evaluation_runner.py` passed the full `retrieved_context` list into the
      Legal RAG judge.
    - `legal_rag_bench_judge.py` joined that full context into the judge prompt
      with no cap.
  - Real artifact audit on the evaluation:
    - 41/100 stored contexts were over 100 KB.
    - 18/100 contained `# Question Seeds`.
    - The remaining 10 judge-error rows all had ~3.2-3.3M char contexts.
  - General Filesystem RAG fixes:
    - `FilesystemRAGTools.read_file()` now refuses full reads over 100 KB and
      tells the agent to use `grep_search`, `headers_only=True`, or line ranges.
    - Filesystem prompt now tells the agent to search `question_seeds.md` with
      `grep_search` rather than read it wholesale.
    - Agent evidence context now excludes `_index/questions/*` and caps stored
      context chunks at 20k chars. Navigation indexes can still guide the agent,
      but they are not stored as evidence context for downstream evaluators.
  - Legal RAG judge safety fix:
    - `_format_judge_context()` filters navigation/question-seed chunks,
      truncates each evidence chunk to 8k chars, and caps total judge context at
      40k chars.
    - Verified against the 10 contaminated rows: each ~3.3M char retrieved
      context became <=40k chars, and `# Question Seeds` was absent.
  - Regression tests added for oversized full-read rejection, question-seed
    evidence exclusion, and judge prompt context filtering/capping.
  - Verification passed:
    py_compile, ruff, backend judge tests, and filesystem agent tests. Root
    pytest emitted a cache permission warning but no test failure.

- Rejudged the full affected Filesystem Legal RAG evaluation with sanitized
  context (2026-07-02):
  - Extended `repair_legal_rag_judge_errors.py` with `--all`, which rejudges
    every Legal RAG row instead of only parse errors / missing taxonomy.
  - Added script helper tests in
    `platform/backend/tests/test_scripts/test_repair_legal_rag_judge_errors.py`.
  - Dry-run behavior:
    - default mode selected the remaining 10 `judge_error` rows,
    - `--all` selected all 100 Legal RAG rows.
  - Ran:
    `uv run python scripts/repair_legal_rag_judge_errors.py 4d5e5ad0-66bb-461d-8a4d-f65f548b5058 --all --timeout-seconds 300`.
  - The run completed and committed. Final DB state:
    - `count=100`
    - `judge.count=100`
    - `judge.scored_count=100`
    - `judge.parse_error_count=0`
    - `classified_count=100`
    - `correct_rate=0.72`
    - `grounded_rate=0.72`
    - `retrieval.gold_accessed_rate=0.59`
  - Final taxonomy:
    - `success=62`
    - `hallucination_or_ungrounded=26`
    - `retrieval_error=8`
    - `reasoning_error=2`
    - `abstention=2`
  - Final repair dry-run returned 0 rows.

- Ported the general Filesystem RAG safety fixes to `main` (2026-07-02):
  - Main worktree: `C:\tmp\RAG-evaluator-main-fix`, branch `main`.
  - `main` already had unrelated dirty changes; only the general files were
    patched, preserving the existing transient-provider retry changes there.
  - Ported:
    - `agent/tools.py`: 100 KB full-read guard.
    - `agent/prompts.py`: search `question_seeds.md` via `grep_search`.
    - `agent/agent.py`: exclude `_index/questions/*` from evidence context and
      cap context chunks at 20k chars.
    - `tests/unit/test_filesystem_rag_agent.py`: regression coverage.
  - Not ported to `main`: `legal_rag_bench_judge.py` sanitizer, because that
    Legal RAG backend service does not exist on `main`.
  - Verification in the `main` worktree passed:
    py_compile, ruff, and `tests/unit/test_filesystem_rag_agent.py` (6 passed).

- Analyzed the failed-28 report and wrote the Filesystem RAG improvement plan
  (2026-07-03):
  - Input: `reports/filesystem_rag_failed_28_analysis.md` (28 binary-incorrect
    cases from `4d5e5ad0...`; 20 gold-missed, 8 gold-accessed-but-failed;
    failed-only rerun recovered 10/27 purely from run-to-run variance).
  - Output: `reports/filesystem_rag_improvement_plan.md` (local, `reports/` is
    gitignored). 20 findings across agent, preparation pipeline, and platform,
    each with code references, plus a prioritized roadmap.
  - Top code findings from the review:
    - `format_tool_result` truncated EVERY tool result at 2,000 chars and
      JSON-encoded `read_file` output, so the agent saw ~1.5k chars of any
      document it read while judge-facing context kept 20k chars per chunk.
      Likely explains most gold-accessed-but-failed cases.
    - The lexical prefetch expansions/weights in `agent/agent.py` are hand-tuned
      to specific benchmark questions (overfitting); plan replaces them with a
      BM25 passage index built at preparation time.
    - `is_reasoning_model()` hardcoded `deepseek-v4-flash`, so temperature was
      silently dropped in the agent loop AND the LiteLLM judge path; provider
      default temperature applied -> the observed rerun variance.
    - Preparation heuristics are tech-corpus-hardcoded (topics
      technical/business/science/general, entities ChromaDB/Google), question
      seeds are templates, token usage is fabricated estimates, `generate()`
      re-runs the whole agent.

- Committed the outstanding WIP as a clean baseline (2026-07-03):
  - 5 commits split by area: core FS evidence bounds (`c7c2453`), backend judge
    hardening + partial-failure metrics (`4050d68`), frontend judge-error/
    partial-results UI (`2f4b0bd`), article docs (`5f521ba`), permission
    allowlist chore (`e08aea6`). `prompt-example.txt` left untracked (scratch).

- Roadmap step 1: raised the tool-result budget so the agent sees the evidence
  it reads (2026-07-03, commit `fea4212`):
  - `agent/prompts.py`: `read_file` results are now rendered as plain text with
    a one-line scope header (`[partial read; file has N lines]`) instead of
    JSON, with a 10,000-char budget; `grep_search` gets 6,000; navigation tools
    (`list_directory`, `find_files`, `get_file_info`) keep 2,000. Truncated
    read_file results tell the agent to re-read with `start_line`/`end_line`.
  - No call-site changes; per-tool defaults live in `TOOL_RESULT_LIMITS`.
  - Verified: 3 new unit tests (plain-text rendering, re-read hint, navigation
    cap); all tests in `tests/unit/test_filesystem_rag_agent.py` pass; ruff and
    mypy clean.

- Roadmap step 2: final-answer contract and refusal retry (2026-07-03, commit
  `36c0fa5`):
  - `agent/prompts.py`: replaced the one-line Response Format with a Corpus
    Context section (legal educational material; answer sensitive sexual-offence
    questions neutrally; name the missing legal element instead of refusing) and
    an Answer Contract (English; first sentence states the conclusion; preserve
    material qualifiers; no uncited statutes/cases; proportional length).
    Compact prompt got a condensed version. Added `format_answer_retry_prompt`.
  - `agent/agent.py`: `unusable_answer_reason()` classifies final answers as
    `empty`, `non_english` (CJK-dominant), or `refusal` (conservative opening
    patterns only). Flagged answers get ONE corrective retry reusing the
    gathered conversation without tools (`_retry_unusable_answer`). Applied in
    both the normal finish and the max-iterations synthesis path. Metadata
    records `answer_retries`/`answer_retry_reason`.
  - Targets the 4 sexual-offence refusal/empty cases and the qualifier-dropping
    essay answers from the failed-28 report.
  - Verified: classifier + end-to-end fake-client retry test + prompt content
    test; 13 tests pass; ruff and mypy clean.

- Roadmap step 3: determinism - stop dropping temperature for deepseek and
  record request params (2026-07-03, commit `1caf39d`):
  - `src/rag_evaluator/common/llm_utils.py`: split `rejects_temperature()`
    (o-series and gpt-5 only) out of `is_reasoning_model()` (which keeps
    deepseek-v4-flash for reasoning_effort forwarding, catalog capability flags,
    and RLM token budgets). `get_safe_llm_params` drops temperature only for
    models that reject it, so `deepseek-v4-flash` now gets `temperature=0.0` by
    default instead of the provider default (~0.7-1.0) that caused 10/27 rerun
    flips. Also fixed a pre-existing mypy arg-type in
    `is_transient_llm_error`.
  - `platform/backend/app/services/llm_provider.py`: same split applied to the
    LiteLLM path - this is also the JUDGE path, so the judge was running at
    provider-default temperature for deepseek despite passing `temperature=0`.
    The existing runtime fallback (retry without temperature on provider error)
    keeps this safe for unknown models.
  - `agent/agent.py` + `filesystem_rag.py`: query metadata now carries
    `llm_request_params` (model, temperature, reasoning_effort; None =
    omitted/provider default) plus the step-2 retry fields through both
    `query()` and `query_with_trace()`, so stored runs document what was
    actually sent.
  - Verified: 18 core unit tests pass (new `rejects_temperature` +
    deepseek-keeps-temperature coverage, agent metadata assertion); 12 backend
    tests pass including a new one asserting deepseek sends `temperature=0` to
    litellm while gpt-5 sends None; ruff and mypy clean.

- Implemented roadmap steps 4-8, completing the full improvement plan
  (2026-07-03, committed together with the round-2 fixes below):
  - Step 4 (A2/B2/B3): new `passage_index.py` builds an Okapi BM25
    (`k1=1.5, b=0.75`) passage index at preparation time, persisted to
    `_index/passages/bm25.json` via `index_builder.py:build_all_indexes()`.
    New `agent/prefetch.py` + `tools.py` `search_passages()` tool replace the
    old hand-tuned prefetch. The old `_extract_prefetch_terms`/
    `_prefetch_term_weight` benchmark-specific term-weight functions in
    `agent/agent.py` are deleted outright (zero remaining matches in `src/`),
    not merely extracted; `agent.py` shrank from ~850 to 676 lines.
    LLM summary/question-seed prompts in `analyzer.py` were strengthened to
    preserve decisive rules/qualifiers, and `use_llm_synthesis`/
    `force_analysis_method` are now real config knobs
    (`filesystem_use_llm_synthesis`, `filesystem_force_analysis_method`)
    instead of hardcoded off. Gap: the heuristic (default) summary and
    question-seed generators themselves are unchanged - only the opt-in LLM
    path improved.
  - Step 5 (A6/A10): `grep_search()` now ranks matches by relevance and
    reports `total_matches`/`returned_matches`/`truncated`. The agent loop
    short-circuits into `_synthesize_partial_answer()` the instant a
    tool-call/file-read limit is hit, instead of idling through remaining
    `max_iterations`.
  - Step 6 (A8/A9): token accounting now comes from `response.usage` on every
    chat completion (`agent.py` `_record_llm_usage`), flowing through
    `filesystem_rag.py` to `evaluation_runner.py`; the fabricated
    character-based estimate is deleted. `filesystem_rag.py:_generate_only()`
    now issues one plain chat completion instead of re-running the full agent
    loop.
  - Step 7 (C1/C3): new `derive_success_signals()` in
    `legal_rag_bench_metrics.py` splits `g_eval_pass`/`judge_correct`/
    `judge_grounded`/`taxonomy_success`/`gold_accessed` into separate fields
    and adds `alternate_evidence_supported`/`correct_without_gold` (credits
    grounded-but-gold-missed answers). Exporter headline columns and both
    frontend Legal RAG components (`LegalRagBenchMetrics.tsx`,
    `LegalRagBenchComparison.tsx`) updated to show the split signals.
  - Step 8 (B1/B4/C6/C5): analyzer topic/entity keywords swapped from
    tech-corpus to legal-corpus terms (still hardcoded, just retargeted, not
    truly corpus-adaptive outside the LLM path); document ids now derive from
    passage ids embedded in source filenames when present (B4); unused
    prompt/prefetch exports deleted (C6); `evaluation_runner.py` deepeval
    imports made lazy, removing that specific pytest-collection-hang cause
    (C5; other import-time hangs, e.g. chromadb, untouched). Roadmap item A11
    (router) was not addressed - `router.py` still carries the tech-corpus
    term list, only its hint text was updated.
  - New/expanded test coverage: `tests/unit/test_filesystem_rag_agent.py`
    (+536 lines) and `tests/unit/test_index_builder.py` (+51 lines) cover
    BM25 prefetch, the `search_passages` tool, grep ranking/truncation,
    limit short-circuiting, real token usage, passage-id filename
    resolution, and single-call `generate()`; backend
    `test_legal_rag_bench_metrics.py`/`test_evaluation_exporter.py` cover the
    split success signals and alternate-evidence credit.
  - Status update (later 2026-07-03 session): tests were run and everything
    was committed together with the round-2 fixes below. The failed-28 subset
    has still not been rerun. `prompt-example.txt` (untracked, 3.3MB) is an
    unrelated manual debugging dump, left alone.

- Ran the 28-case clean re-eval and implemented the round-2 improvement plan
  (2026-07-03):
  - Input eval: `Legal RAG clean - filesystem - 2026-07-03 12:15`
    (`a870ea4306eb4b5798c9703d82748dfe`, 28 previously-failed cases, DeepSeek):
    `gold_accessed` 14/28 (50%), judge-correct 11/28. Failure analysis + plan:
    `reports/filesystem_rag_improvement_plan_round2.md` (local, gitignored).
    Root causes: BM25 top-k crowd-out (one doc's section windows filling all
    slots), sibling-chunk satisficing (gold unread in the agent's own tool
    output in 5 cases), vocabulary gap (benchmark wording vs statute-book
    wording, echo summaries), and a DeepSeek DSML tool-call-markup leak
    returned as a final answer.
  - Implemented plan Fixes 1-7 (Fix 8, the prep-time concept-alias index, was
    deliberately skipped - it requires corpus re-preparation approval):
    - Fix 1 `passage_index.py`: `search()` dedupes ranked results to the best
      section window per `doc_id` and reports suppressed windows as
      `other_matching_sections`; prefetch candidates bumped 3 -> 5
      (`agent.py:_build_prefetch_context`).
    - Fix 2 `agent/agent.py` + `prompts.py`: `unusable_answer_reason()` now
      returns `tool_markup` (DSML/invoke-whitelist regex, fullwidth-bar safe);
      the loop gives ONE chance to re-issue the intended action as a real tool
      call (`TOOL_MARKUP_RETRY_PROMPT`, `markup_recovery_used` metadata),
      falling back to the plain-completion retry when budgets are exhausted.
    - Fix 3 `agent/tools.py` + `prompts.py`: reading `documents/<id>.md` now
      returns `section_siblings` (same-section chunks with first informative
      header, noise-header `# <hex8> Passage NNNN` skipped, cap 40, per-section
      cache). `format_tool_result` reserves budget so the sibling block
      survives content truncation.
    - Fix 4 `prompts.py`: strategy rules 9 (search with question vocabulary AND
      statutory paraphrase; zero grep hits are not proof of absence) and 10
      (bench-notes vs charge-book siblings; check the sibling list before
      finalizing).
    - Fix 5 `agent/agent.py`: one-shot evidence nudge - a usable final answer
      built on <2 distinct `documents/` reads triggers a single verification
      request (`format_evidence_nudge_prompt`, `evidence_nudge_used` metadata)
      when budgets allow, then the next answer is accepted either way.
    - Fix 6 `agent/tools.py`: `grep_search` refactored into `_grep_once` +
      orchestration; a plain multi-word pattern with 0 matches transparently
      re-runs in AND-mode and marks `"fallback": "match_all_terms"`. Deviation
      from plan: used an explicit regex-metacharacter-set check instead of the
      suggested `re.escape(p) == p`, which can never pass on Python 3.12
      (escapes spaces).
    - Fix 7 `filesystem_rag.py`: `_reportable_sources`/`_is_passage_source`
      filter `_index/`/`.meta.json` noise from `files_read`/`context_sources`
      metadata (keep meta.json-resolved or passage-stem-matching entries; fall
      back to unfiltered when the filter empties the list, so `doc_NNN`
      corpora still report sources). `_track_agent_response` perf counter
      untouched.
  - Verified: 49 unit tests pass (13 new) in `test_filesystem_rag_agent.py` +
    `test_index_builder.py`; `test_filesystem_preparation.py` integration
    passes; ruff + mypy clean. Read-only replay against the real corpus
    (`idx_46d32975669e4340bc6d031f`): provocation gold `8.12-c3-s1` rank 7 ->
    4 among 10 distinct docs; "jury room experiments" -> `1.5-c5-s1` rank 1;
    alibi gold `3.6-c1-s1` rank 3; paraphrase "punished more than once same
    act" -> `3.12-c1-s1` rank 1; `8.12-c2-s1` sibling map lists `8.12-c3-s1`
    titled "Heat of passion".
  - Known write-offs (plan section 4): `7.3.13.5-c4-s6` (near-duplicate charge
    scripts across offences, benchmark-hard), `4.23-c2-s1` (correct via
    duplicate passage; `alternate_evidence_supported` already credits it),
    `3.6-c1-s1` (misleading alibi framing; expect improvement, not certainty).

## Branch code review and fixes (2026-07-04)

Ran a full code review of the `legal-rag-bench` branch diff vs `main` (8 finder
angles + verification). 8 findings, all fixed and verified (72 backend tests
pass, frontend `tsc` clean, ruff clean):

1. `llm_provider.py` - the strip-temperature retry consumed a loop attempt; on
   the last attempt the real error was masked by a generic RuntimeError. Now an
   explicit counter; temperature retry no longer consumes an attempt.
2. `repair_legal_rag_judge_errors.py` - rejudging updated `judge`/`taxonomy`
   but left `success_signals` stale, and the recomputed summary aggregated the
   stale signals. Now recomputes via `derive_success_signals` (preserves stored
   `g_eval_score`, uses `settings.EVAL_G_EVAL_THRESHOLD`).
3. Exporter `TAXONOMY_ORDER` and `LegalRagBenchComparison.tsx` taxonomy rows
   omitted the `grounded_but_incorrect` bucket - those cases silently vanished
   from taxonomy tables. Bucket added to both.
4. `filesystem_rag.py` resumable prepare - `all([])` was vacuously true, so a
   source with no prepared artifacts was checkpointed complete. Unmapped
   sources now hit `fail_document`.
5. `legal_rag_bench_judge.py` - `_parse_judge_json` crashed with
   AttributeError when the judge returned valid non-object JSON (array/string),
   bypassing the parse-error path. Added an isinstance guard.
6. + 7. N+1 artifact fetches in comparison JSONL export and in
   `_collect_legal_rag_summary` - added `ArtifactStore.retrieve_json_by_ids`
   (single `IN()` query) and used it at both sites.
8. `agent.py` - merged the two copy-paste limit blocks into one parameterized
   guard and threaded `markup_recovery_used`/`evidence_nudge_used` into the
   partial-answer metadata (they were silently dropped on limit-hit answers).

Deliberately skipped (design debt, needs its own change): parallelizing the
per-case judge + DeepEval metric calls; judge's byte-identical parse retry;
BM25 snippet file re-reads per query; passage-id extraction heuristics (should
become a structured `passage_id` trace contract); `filesystem_rag` string
special-casing in metrics (should be a registry capability flag); model-name
substring lists in `llm_utils`.

## Current Step

- Round-1 roadmap (steps 1-8) AND round-2 plan Fixes 1-7 are implemented,
  tested, and committed on `legal-rag-bench` (steps 1-3: `fea4212`, `36c0fa5`,
  `1caf39d`; steps 4-8 + round 2: this session's commits). Round-2 Fix 8
  (concept-alias index) is deliberately deferred - it needs corpus re-prep
  approval.
- Improvement plans: `reports/filesystem_rag_improvement_plan.md` and
  `reports/filesystem_rag_improvement_plan_round2.md` (local, gitignored).
- Branch strategy decided: no permanent legal branch. `main` has zero commits
  the branch lacks, so once article work stabilizes, merge `legal-rag-bench`
  into `main`, delete the branch, and work main-first from then on. The
  benchmark services are product code (a benchmark profile), not customization.

## Pending

- Rerun the 28-case clean subset from the web UI with the SAME model settings
  as `a870ea4306eb4b5798c9703d82748dfe` and compare `gold_accessed_rate`
  (baseline 0.50, target >=0.75 without Fix 8) and judge-correct (baseline
  11/28). Do NOT re-prepare the corpus (re-prep changes the KB index id)
  unless round-2 Fix 8 is explicitly approved.
- Decide on round-2 Fix 8 (prep-time concept-alias index + BM25 alias
  folding, gated behind `filesystem_use_llm_synthesis`): requires re-prep
  (~8.5 min + one LLM call per section) and a new KB index.
- Re-preparation is required for the BM25 index (`_index/passages/bm25.json`)
  to exist on any prepared corpus that predates roadmap step 4.
- Manual UI refresh/check for `Legal RAG clean - filesystem - 28 giu, 22:35` to
  confirm the results page shows 100 classified Legal RAG rows and no judge
  errors after the DB rejudge.
- Decide whether to rejudge other historical Filesystem Legal RAG evaluations
  whose stored `retrieved_context` artifacts contain `# Question Seeds` or
  oversized context.
- Verify UI can import Legal RAG Bench passages as a Knowledge Base (manual UI).
- Perform Phase 0 UI smoke once product support is ready (manual UI).
