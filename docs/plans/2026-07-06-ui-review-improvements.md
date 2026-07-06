# UI Review and Improvement Plan (2026-07-06)

Review of the platform frontend (`platform/frontend/src`) plus the backend list endpoints that back it.
Findings are ordered by phase; each phase is a small, independently shippable increment.

## Key facts discovered during review

- All backend list endpoints share `Pagination` (`platform/backend/app/api/deps.py`): default `limit=20`, max `100`.
  Most frontend list calls pass no params, so **every list silently truncates at 20 items** with no
  pagination UI (evaluations, KBs, test sets, RAG configs, comparisons).
- `list_evaluations` (`platform/backend/app/api/evaluations.py:631`) calls `_evaluation_to_response(e)`
  without `result_count`, so the Evaluations tab shows **"0 results" on every row**, and the
  partial-results logic in `EvaluationsTab` (which reads `result_count` from the list) never triggers.
- `EvaluationResults.tsx:56` has `const [page] = useState(1)` — page is hardcoded, only the first 50
  results are ever shown, and the search box filters only that first page.
- KB detail embeds ALL documents in `GET /knowledge-bases/{kb_id}` (no paginated documents endpoint
  exists); test set detail likewise embeds all test cases.
- `EvaluationsTab` in `ProjectDetail.tsx` contains a dead in-tab detail view: `setActiveEvaluationId`
  is only ever called with `null` (rows navigate to the `/projects/:id/evaluations/:id` route instead).
- Deleting a document in `KBDetail.tsx` has **no confirmation** (every other destructive action uses
  `window.confirm`).
- The evaluation list response has `test_set_id` / `knowledge_base_index_id` but no names, and the
  index list response has `project_id` but no `project_name` — so neither list can display
  human-readable context.
- `Indexes.tsx` (global page) is stylistically inconsistent: raw `useState`/`useEffect` instead of
  react-query, hardcoded `gray-*`/`bg-white` colors instead of design tokens, search is client-side
  over a max of 50 fetched rows, and fetch errors render as "No indexes found".

## Phase 1 — Bug fixes (small, do first)

1. **`result_count` in evaluations list** (backend): in `list_evaluations`, compute per-evaluation
   result counts (single grouped `select(EvaluationResult.evaluation_id, func.count())` over the page's
   ids) and pass them to `_evaluation_to_response`.
2. **Remove dead code**: delete the `activeEvaluationId` branch in `EvaluationsTab`
   (`ProjectDetail.tsx` ~lines 594-659).
3. **Confirm document delete** in `KBDetail.tsx` (match the existing `confirm()` pattern).

## Phase 2 — Evaluations tab: context + filters (user finding 2)

Backend (`evaluations.py`, `schemas/evaluation.py`):
- Add optional denormalized fields to `EvaluationResponse`: `test_set_name`, `index_name`,
  `rag_config_name`, `rag_type` (join `TestSet` and `KnowledgeBaseIndex`; rag info comes from the
  index's `rag_config` / `config_snapshot`). Populate in `list_evaluations` and `get_evaluation`.
- Add filter query params to `list_evaluations`: `test_set_id`, `knowledge_base_index_id`,
  `rag_config_id` (via join on index), keep `status`. Apply to both the page query and count query.

Frontend (`ProjectDetail.tsx` EvaluationsTab, `api/client.ts`):
- Row chips: test set name, index name, RAG type, plus a "Baseline" badge when `is_baseline`.
- Filter bar above the list: dropdowns for test set / RAG config / index / status, driven by the
  already-fetched project test sets, configs, and indexes; filters go into the URL search params
  (same pattern as `Indexes.tsx`) and are passed to the API.

## Phase 3 — Pagination (user finding 3)

1. **Shared component**: `components/ui/PaginationFooter.tsx` — "Showing X–Y of Z", prev/next
   (all `PaginatedList` responses already return `total`, `offset`, `limit`).
2. **Evaluations tab**: page state + `PaginationFooter` (combined with Phase 2 filters).
3. **Evaluation results** (`EvaluationResults.tsx`): wire the existing `page` state to real controls.
   Optionally add a `search` query param to `GET /evaluations/{id}/results` (ILIKE on question /
   answers via the joined test case) so search spans all pages instead of the loaded page.
   Note: `DifficultyChart` currently aggregates the loaded page only — acceptable short-term; the
   proper fix is a small server-side per-difficulty aggregate endpoint (optional, later).
4. **KB documents**: new endpoint `GET /knowledge-bases/{kb_id}/documents` (Pagination + optional
   `search` on filename). `KBDetail.tsx` switches to it with `PaginationFooter` + a search box.
   Then stop embedding `documents` in the KB detail response (check other consumers first);
   the KB response already carries `document_count`.
5. **Test cases**: same pattern — `GET /test-sets/{id}/cases` paginated; `TestSetDetail.tsx` uses it.
   Lower priority (test sets are typically ~50-200 cases, but generation can produce more).

## Phase 4 — Global Indexes page (user finding 1)

Verdict: the page is useful as a cross-project "build status" console (and it is the only place to
watch builds across projects), but as-is it doesn't show which project an index belongs to, which is
what makes it feel wrong.

- Backend: add `project_name` to the index list response (it already denormalizes
  `knowledge_base_name` and `rag_config_name`).
- Frontend `Indexes.tsx`:
  - Show project name on each card (link to the project) and add a project filter dropdown.
  - Migrate to react-query + theme design tokens (remove `gray-*`/`bg-white`).
  - Proper error state (currently errors look like an empty list).
  - Pagination via `PaginationFooter` instead of the fixed `limit: 50`.
- Alternative considered: remove the sidebar entry entirely (project tab + Playground cover most
  uses). Not recommended — cheap to fix, and it's the only cross-project build monitor.

## Phase 5 — Polish (optional, as time allows)

- Surface API errors as toasts (the axios interceptor only `console.error`s today).
- Replace scattered `window.confirm` with a shared `ConfirmDialog` for visual consistency.
- Pass-rate color scale: add a red band for low values (currently green >= 70%, amber otherwise).

## Verification

- Backend: run the API and hit endpoints directly (pytest hangs on this machine — use
  `dev_server.py` + curl or direct imports).
- Frontend: `npm run lint` + `npx tsc --noEmit` in `platform/frontend`, then manual walkthrough:
  project with >20 evaluations, KB with >100 documents, evaluation with >50 results.
