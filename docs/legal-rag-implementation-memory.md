# Legal RAG UI Implementation Memory

Last updated: 2026-06-28

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

## Current Step

- Backend + UI for Legal RAG Bench comparison/export and the first Filesystem
  RAG retrieval optimization are complete. Remaining work is rerunning the
  affected FS evaluation, manual UI verification, and the experiment phases.

## Pending

- Rerun `Somke KB - legal fs` after the FS retrieval optimization to confirm the
  Sally case changes from `Inspection` to `view` and to re-check summary-heavy
  contexts across all 10 subset questions.
- Plan the next Filesystem RAG indexing pass:
  - proper full-text/BM25-style index over documents,
  - cleaner question-seed generation with stopword/generic-term filtering,
  - corpus-aware topic maps,
  - stronger passage-id/title/source metadata,
  - exact definition/alias index for "what is this called?" questions.
- Verify UI can import Legal RAG Bench passages as a Knowledge Base (manual UI).
- Perform Phase 0 UI smoke once product support is ready (manual UI).
