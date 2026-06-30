# Legal RAG Bench → `main` Port **Status** Report

**Date:** 2026-06-30
**Author:** branch/main reconciliation pass
**Compared:** `main` (`5ee089a`) vs `legal-rag-bench` (`6877a66` + uncommitted WIP), and the intermediate `port/runner-and-rag-fixes` (`15f5fd4`)

This report supersedes the *plan* in `docs/legal-rag-bench-main-port-analysis.md`
(written 2026-06-27, analyzing only commit `3b80126`). That document said *what
should* be ported. This document says *what is actually in each branch right
now*, because work continued on `legal-rag-bench` after the plan was written.

---

## ✅ EXECUTION STATUS (updated 2026-06-30)

The plan in this report has been **executed**. Summary of the final state:

**Step 1 — WIP committed on `legal-rag-bench`** (tree is now clean):
- `708d31a` resume + orphaned-eval reconcile · `e4a5767` FS retrieval opt ·
  `6a7429f` pricing defaults (the three **general** commits)
- `ef4f7d7` article docs + `.gitignore` · `a985ba7` local settings (branch-only)
- `data/legal_rag_bench_clean/` (20 MB, regenerable) and `.serena/` were
  gitignored, not committed.

**Step 2 — all general fixes ported to `port/general-fixes-round-2`** (branched
off `main`, **6 commits ahead**, contains **no legal-only files**; supersedes the
older `port/runner-and-rag-fixes`):
- `15f5fd4` items 1–6 · `88b0d2c` B1 retries · `9c3bf54` B2 retry UI ·
  `44615e0` B3 reconcile+resume · `bcfd15a` B4 FS prefetch · `7d18ce1` B5 pricing.
- B3 was the only merge conflict: kept the general resume helpers
  (`_get_completed_test_case_ids`, `_has_result_for_test_case`), dropped the
  legal `_collect_legal_rag_summary`.

**Verification:** backend ruff ✅ · root ruff ✅ · frontend lint ✅ ·
frontend build/tsc ✅ · 36 targeted backend+FS tests ✅. Only
`test_api/test_evaluations.py` could not run (documented `deepeval`/`chromadb`
collection hang — not a regression; it passed on `legal-rag-bench`).

**Decisions:** optional items **B6** (FS `gold_accessed` resolution) and **B7**
(test-case `metadata` column + migration) were **skipped** — they stay on
`legal-rag-bench`.

**Remaining (owner action):** review and merge `port/general-fixes-round-2` into
`main` (`git merge --no-ff port/general-fixes-round-2`), then optionally delete
both `port/*` branches.

---

## 0. The one thing to know first

> **`main` currently contains NONE of the legal-rag-bench fixes.**
> The porting work was done on a side branch `port/runner-and-rag-fixes`
> (single commit `15f5fd4`) that was **never merged into `main`**. `main` is
> still sitting exactly at the branch point (`5ee089a`).

So "what has been ported to main" = nothing is *in* `main` yet, but items 1–6
of the original plan are **staged and verified** on `port/runner-and-rag-fixes`,
ready to merge. Everything created after 2026-06-27 is **not ported anywhere**.

### Branch topology

```
5ee089a (main, HEAD of main)  ← branch point, nothing ported in
   │
   ├── port/runner-and-rag-fixes
   │      15f5fd4  "Port general fixes..."   ← items 1–6, clean, UNMERGED
   │
   └── legal-rag-bench
          3b80126  big Legal RAG Bench commit (analyzed by the old plan)
          9a89e6d  Fix hit@k undercount        (legal-only)
          c3443d5  FS gold_accessed resolution (borderline)
          854c415  Add evaluation retry UI     (GENERAL — not ported)
          f78cb84  Bound LLM completion retries (GENERAL — not ported)
          6877a66  Legal RAG Bench dataset notice (article-only)
          + UNCOMMITTED WIP: resume/reconcile, FS retrieval opt, pricing
```

---

## 1. Category A — Staged on `port/runner-and-rag-fixes`, ready to merge

These six are the original plan's "port to main" list. I verified each is
present on the port branch and that the legal-only code was correctly stripped
(`evaluation_runner.py` on the port branch has **zero** `legal_rag` references).

| # | Change | Files | Verified |
|---|--------|-------|----------|
| 1 | Per-test-case metric isolation (concurrency bug) | `evaluation_runner.py` (+ `test_evaluation_runner.py`) | `_initialize_metrics` called per case; `test_async_runner_uses_isolated_metric_instances` present |
| 2 | Full db-lock coverage in async runner | `evaluation_runner.py` | artifact stores + commit + progress + checkpoint all under `_db_lock` |
| 3 | Checkpoint `last_index` = real completed count | `evaluation_runner.py` | uses `_get_completed_count()` |
| 4 | Hybrid RAG embedding dimension (3072 for `text-embedding-3-large`) | `hybrid_rag.py`, `registry.py` | `embedding_dimension` build param on port, absent on `main` |
| 5 | Batched multipart upload (>1000 parts) | `client.ts` | present |
| 6 | Anti-over-refusal answer prompt (all 4 RAGs) | `chroma_rag.py`, `hybrid_rag.py`, `neo4j_rag.py`, `google_vertex_rag.py` | new prompt on port + branch, old prompt on `main` |

**Action:** verify (`platform/backend` pytest + ruff, frontend lint) then merge
`port/runner-and-rag-fixes` into `main`. This is the lowest-risk, highest-value
step and clears the bulk of the backlog in one move.

---

## 2. Category B — GENERAL fixes MISSING from both `main` and the port branch

These were created on `legal-rag-bench` **after** the 2026-06-27 port plan, so
the port branch never saw them. They are genuinely useful to the product and
should be ported. Two are committed; the rest are uncommitted WIP and must be
committed (or applied) first.

### B1. Bound LLM completion retries — commit `f78cb84` — PORT (clean)
- Files: `platform/backend/app/config.py`, `app/services/llm_provider.py`,
  `src/rag_evaluator/common/llm_utils.py`, `tests/test_services/test_llm_provider.py`.
- Self-contained, no legal coupling, has tests. Caps retry attempts so a failing
  provider can't spin forever. **Port as-is.**

### B2. Evaluation retry UI — commit `854c415` — PORT (general feature)
- Files: `app/api/evaluations.py`, `app/services/job_event_log.py`,
  `frontend/src/api/client.ts`, `EvaluationProgress.tsx`,
  `hooks/useEvaluationStream.ts`, `pages/EvaluationDetail.tsx`,
  `pages/ProjectDetail.tsx`.
- Adds a Retry button for failed/cancelled/partial evaluations, SSE reconnect,
  persisted error/count after refresh, and a job-event replay-cache clear on
  retry. No legal coupling. **Port the whole commit.** (Only the doc-memory hunk
  in it is article-related and can be dropped.)

### B3. Robust evaluation resume + orphaned-eval reconciliation — UNCOMMITTED WIP — COMMIT then PORT
A coherent general reliability feature, currently uncommitted on the working
tree, backed by `docs/plans/2026-06-29-reconcile-orphaned-evaluations.md`:
- `app/services/job_checkpoint_service.py` — new `reconcile_orphaned_evaluations()`
  (marks `running`/`pending` evals as recoverable `failed` on startup; mirrors
  `reconcile_interrupted_builds`). No DB migration.
- `app/main.py` — calls it in the lifespan startup.
- `app/services/evaluation_runner.py` — resume by saved-result ids, per-case
  error tracking, partial-completion failure signalling.
- `app/services/job_event_log.py` — persisted event replay after refresh + reset
  on retry.
- `EvaluationProgress.tsx` — follow-up UI tweaks on top of B2.
- Tests: `test_evaluation_lifecycle.py`, `test_job_checkpoint_service.py`,
  `test_job_event_log.py`, `test_api/test_evaluations.py`.
- **All general.** Depends on B2 (shares `job_event_log.py` / `EvaluationProgress.tsx`),
  so port B2 first, then this. **Must be committed on `legal-rag-bench` first**
  (it is WIP right now).

### B4. Filesystem RAG retrieval optimization — UNCOMMITTED WIP — PORT (mechanism), review constants
- Files: `filesystem_rag/agent/agent.py` (+148), `agent/prompts.py` (+11),
  `tests/unit/test_filesystem_rag_agent.py` (+86).
- General mechanism: deterministic lexical prefetch with idf/title weighting and
  **full-document excerpt injection** for top candidates (helps any corpus where
  summaries are lossy / exact wording matters).
- Caveat: the term weights/synonyms (`view`, `inspection`, VARE, etc.) are
  **legal-benchmark tuned**. Port the mechanism; either keep the weights as a
  pragmatic default or make them corpus-adaptive before porting. Uncommitted.

### B5. Pricing defaults refresh — UNCOMMITTED WIP — PORT (low risk, optional)
- File: `app/utils/pricing_defaults.py`. Adds OpenRouter model prices
  (`deepseek/*`, `openai/gpt-5.4*`, `gpt-5.5`) and fixes a stale comment.
  General data update, but model selection leans toward the article's runs.
  **Port if you want those models priced in the cost view.** Uncommitted.

### B6. FS `gold_accessed` doc→passage resolution — commit `c3443d5` — BORDERLINE / optional
- File: `src/rag_evaluator/rag_implementations/filesystem_rag/filesystem_rag.py`.
- Resolves `doc_NNN`/`doc_NNN_summary` sources back to the original passage file
  via `documents/doc_NNN.meta.json` `original_file`. Built to make the legal
  `gold_accessed` metric match, but it generally improves FS **source
  provenance** (reports real source filenames instead of synthetic doc ids).
- **Port only if** you want original-file provenance in FS source reporting;
  otherwise it's harmless to leave on the branch. (The metric that *consumes* it
  lives in `legal_rag_bench_metrics.py`, which is legal-only — see C.)

### B7. Generic `metadata` column on test cases — part of `3b80126` — OPTIONAL (needs migration)
- Files: `app/models/test_case.py`, Alembic
  `20260626_000001_add_test_case_metadata.py`, `app/schemas/test_set.py`,
  `app/api/test_sets.py`, `ImportTestSetDialog.tsx`, `client.ts` (`TestCase` /
  `TestCaseCreate`).
- Generic JSON column (mapped `metadata_`), plumbed through create/bulk/update/
  import/export. Reusable, but added to carry legal fields. The port branch
  **deliberately skipped it.** **Port if wanted — the Alembic migration must go
  with it.**

---

## 3. Category C — Legal / article ONLY — keep on `legal-rag-bench`, do NOT port

### New backend services (legal-only files — do not exist in `main`)
- `app/services/legal_rag_bench_judge.py` — binary LLM judge.
- `app/services/legal_rag_bench_metrics.py` — hit@k / retrieval / taxonomy /
  `summarize_legal_rag_metrics` (**includes commit `9a89e6d`'s hit@k undercount
  fix** — a real bug fix, but in a legal-only file, so it cannot be ported to
  `main` without the whole service).
- `app/services/evaluation_exporter.py` — article export (Markdown/CSV/JSONL).
- Tests: `test_legal_rag_bench_judge.py`, `test_legal_rag_bench_metrics.py`,
  `test_evaluation_exporter.py`.

### Legal hunks inside shared files (strip when porting, keep on branch)
- `evaluation_runner.py` — legal judge init, `legal_rag_result` block,
  `raw_metrics["legal_rag_bench"]`, `_collect_legal_rag_summary`, finalize
  summary. (Already correctly excluded from the port branch.)
- `app/api/comparisons.py` — `/comparisons/{id}/export` endpoint + export helpers.
- `app/api/evaluations.py` — `GET /evaluations/{id}/raw-metrics/{result_id}`
  (generic-ish; low-priority debugging aid — leave unless you want it).
- `app/schemas/evaluation.py` — `SummaryMetrics.legal_rag_bench` field.

### Frontend (legal-only)
- New: `components/comparisons/LegalRagBenchComparison.tsx`,
  `components/evaluations/LegalRagBenchMetrics.tsx`.
- Legal hunks in: `ComparisonDetail.tsx`, `compare-utils.ts`,
  `EvaluationResults.tsx` (`getSummaryScore`), `StartEvaluationWizard.tsx`
  (`renderMetricOption` + legal metrics group), `api/client.ts` (`getRawMetrics`,
  export URL helper, `legal_rag_bench` fields).

### Scripts, data, docs (article-only)
- `scripts/convert_legal_rag_bench.py` (`clean`/`annotated` modes).
- Commit `6877a66` — `README.md` notice + `data/LEGAL_RAG_BENCH_NOTICE.md`.
- `docs/legal-rag-plan.md`, `docs/legal-rag-implementation-memory.md`,
  `docs/legal-rag-ui-walkthrough.md`, `docs/legal-rag-bench-main-port-analysis.md`,
  `docs/articles/`.
- Untracked working-tree: `data/legal_rag_bench_clean/`, `.serena/`,
  `.claude/settings.local.json`.

> Note: `docs/plans/2026-06-29-reconcile-orphaned-evaluations.md` is a **general**
> plan (drives B3), not article-only — keep it and port the feature it describes.

---

## 4. Recommended action plan (in order)

1. **Merge the staged work.** Verify and merge `port/runner-and-rag-fixes`
   into `main` → lands Category A items 1–6.
2. **Port the two committed general fixes.** Cherry-pick `f78cb84` (B1) and
   `854c415` (B2, drop its doc-memory hunk) onto `main`.
3. **Commit the WIP on `legal-rag-bench`,** splitting general from article:
   - general commit(s): B3 (resume/reconcile), B4 (FS retrieval), B5 (pricing);
   - article commit(s): `data/legal_rag_bench_clean/`, `docs/articles/`, etc.
   Then port the general commits to `main` (B3 after B2).
4. **Optional ports:** B6 (FS provenance), B7 (`metadata` column + migration).
5. **Leave Category C on `legal-rag-bench`.**

### Verification per port (from CLAUDE.md)
- Backend: run from `platform/backend` — `uv run pytest`, `uv run ruff check .`.
  (`uv run pytest` at repo root **hangs on this machine** — deepeval/chromadb
  import at collection; use targeted `-p no:cacheprovider -o addopts="" -q` or
  direct `./.venv/Scripts/python.exe` imports.)
- Frontend: `npm run lint`, `npm run build` from `platform/frontend`.
- For B6 / B7 confirm no Alembic/migration drift; B7's migration must ship.

---

## 5. Quick reference matrix

| Item | In `main`? | On port branch? | On `legal-rag-bench`? | Verdict |
|------|:---------:|:---------------:|:---------------------:|---------|
| 1 Metric isolation | no | yes | yes | merge port |
| 2 db-lock coverage | no | yes | yes | merge port |
| 3 checkpoint last_index | no | yes | yes | merge port |
| 4 Hybrid embed dim | no | yes | yes | merge port |
| 5 Batched upload | no | yes | yes | merge port |
| 6 Anti-refusal prompt | no | yes | yes | merge port |
| B1 Bound LLM retries (`f78cb84`) | no | no | yes (committed) | PORT |
| B2 Retry UI (`854c415`) | no | no | yes (committed) | PORT |
| B3 Resume + reconcile | no | no | yes (WIP) | commit + PORT |
| B4 FS retrieval opt | no | no | yes (WIP) | PORT (review weights) |
| B5 Pricing defaults | no | no | yes (WIP) | PORT (optional) |
| B6 FS gold_accessed resolve (`c3443d5`) | no | no | yes (committed) | optional |
| B7 `metadata` column | no | no | yes (committed) | optional (+migration) |
| C  legal services/UI/scripts/data | no | no | yes | KEEP on branch |
