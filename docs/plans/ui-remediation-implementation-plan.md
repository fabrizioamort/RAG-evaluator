# UI Remediation Implementation Plan

Date: 2026-05-31

Source review: `docs/ui-review.md`

Scope: `platform/frontend`

## Goal

Correct the confirmed UI defects and reshape the frontend around the actual RAG evaluation workflow:

```text
Project
  -> Knowledge Base + RAG Config
  -> Index
  -> Test Set
  -> Evaluation
  -> Results, Comparison, Trends
```

The plan is split so small, low-risk fixes can land first while larger workflow and dialog changes are handled deliberately.

## Implementation Principles

- Keep the product project-centered. Global navigation should not advertise pages that do not exist.
- Fix broken or inert actions before adding new feature surfaces.
- Make every primary workflow reachable from an obvious button, not only from indirect tab navigation.
- Use the existing API client and React Query patterns before introducing new abstractions.
- Standardize dialog behavior once, then refactor existing modals onto that shared component.
- Preserve current working flows while adding route-backed navigation incrementally.

## Phase P0a: Quick Wins

These are small wiring and correctness fixes. They should be implemented before the larger dialog refactor.

### P0a.1 Remove Or Correct Missing Sidebar Routes

Problem:

- `Layout.tsx` links to `/evaluations`, `/trends`, and `/settings`.
- `App.tsx` does not register those routes.
- Clicking those links opens 404.

Files:

- `platform/frontend/src/components/layout/Layout.tsx`
- `platform/frontend/src/App.tsx`

Recommended implementation:

1. Short-term: remove `Evaluations`, `Trends`, and `Settings` from the sidebar.
2. Keep only implemented global pages:
   - Dashboard
   - Projects
   - Indexes
   - Playground
3. Do not add placeholder pages unless they provide real value.
4. Keep project-scoped Evaluations and Trends inside Project Detail for now.

Acceptance criteria:

- Sidebar contains no link that resolves to the 404 route.
- Clicking every sidebar item lands on a functional page.
- Active link styling still works.

Verification:

- Run `npm.cmd run lint`.
- Run `npm.cmd run build`.
- Manually click every sidebar item.

### P0a.2 Fix Sidebar Footer Positioning

Problem:

- Settings is positioned with `absolute bottom-4 left-4 right-4`.
- The parent `aside` is not `relative`.
- This can anchor the footer to the wrong container.

Files:

- `platform/frontend/src/components/layout/Layout.tsx`

Recommended implementation:

1. Make the sidebar a positioned flex column:

   ```tsx
   <aside className="relative flex w-64 flex-col border-r border-border bg-card">
   ```

2. Make the nav area `flex-1`.
3. If Settings is retained later, render it as a normal sidebar footer, not an absolutely positioned block.

Acceptance criteria:

- Sidebar items remain aligned.
- No sidebar footer content overlays the main content.
- The sidebar behaves predictably at desktop heights.

### P0a.3 Fix Broken RAG Config Link From Index Detail

Problem:

- `IndexDetail.tsx` links to `/rag-configs/:id`.
- No route exists for that path.

Files:

- `platform/frontend/src/pages/IndexDetail.tsx`

Recommended implementation:

Short-term:

1. Replace the broken link with a project-scoped navigation target:

   ```text
   /projects/:projectId?tab=rags
   ```

2. If `project_id` is missing, render the config name as plain text.
3. Keep the config snapshot visible on Index Detail so the user is not blocked.

Later:

1. Add a real RAG Config detail route:

   ```text
   /projects/:projectId/rag-configs/:configId
   ```

Acceptance criteria:

- Clicking the RAG config reference never opens 404.
- The user can get back to the owning project's RAG Configs tab.
- Index Detail remains useful even without a config detail page.

### P0a.4 Wire Project Archive And Fix Its Styling Bug

Problem:

- The Project Detail `Archive` button has no `onClick`.
- Its Tailwind classes use backslashes instead of opacity slashes:
  - Incorrect: `bg-destructive\10`
  - Correct: `bg-destructive/10`

Files:

- `platform/frontend/src/pages/ProjectDetail.tsx`
- `platform/frontend/src/api/client.ts`

API already available:

- `api.projects.archive(id)`

Recommended implementation:

1. Add an archive mutation in `ProjectDetail`.
2. Use `api.projects.archive(id)`.
3. Add confirmation before archiving.
4. On success:
   - Invalidate `['project', id]`.
   - Invalidate `['projects']`.
   - Show success toast.
5. On failure:
   - Show error toast.
6. Fix the class names:

   ```tsx
   bg-destructive/10
   border-destructive/20
   hover:bg-destructive/20
   ```

Acceptance criteria:

- Archive button visibly uses destructive styling.
- Clicking Archive asks for confirmation.
- Confirming archives the project and refreshes project data.
- Failed archive calls display an error toast.

Note:

- A shared ConfirmDialog is planned in P3. For P0a, using the existing browser confirm pattern is acceptable if it keeps this change small.

### P0a.5 Fix Global Indexes Search And Run Evaluation

Problem:

- Search input is disabled and has no behavior.
- Run Evaluation from an index card only logs to console.

Files:

- `platform/frontend/src/pages/Indexes.tsx`
- `platform/frontend/src/components/indexes/IndexCard.tsx`
- `platform/frontend/src/pages/IndexDetail.tsx`

Recommended implementation:

1. Implement client-side search while backend search is unavailable.
2. Filter by:
   - index name
   - knowledge base name
   - RAG config name
   - status
3. Enable the search input.
4. Wire `onRunEvaluation` to the same path used by Index Detail:

   ```text
   /projects/:projectId?tab=evals&startEval=1&kbId=:kbId&indexId=:indexId
   ```

5. If an index lacks `project_id`, hide or disable Run Evaluation with a clear tooltip/title.
6. Replace the raw `console.log`.

Acceptance criteria:

- Searching indexes filters visible cards.
- Clearing search restores the full list.
- Ready indexes show a working Run Evaluation action.
- Clicking Run Evaluation opens the owning project with the evaluation wizard prefilled.
- No console-only action remains.

### P0a.6 Narrow The Playground Naming Inconsistency

Problem:

- The only confirmed naming inconsistency is the Playground heading "Select RAG Systems to Compare."
- Other surfaces mostly use "RAG Config" or "Index."

File:

- `platform/frontend/src/pages/Playground.tsx`

Recommended implementation:

Change the heading to one of:

- "Select Indexes to Compare"
- "Select RAG Indexes to Compare"

Preferred:

```text
Select Indexes to Compare
```

Acceptance criteria:

- Playground copy matches the product model.
- It is clear that users are querying built indexes, not editing reusable RAG configs.

## Phase P0b: Shared Dialog Refactor

This is intentionally separate from P0a. The review identifies a real defect, but the fix touches many components and should be handled as a focused refactor.

### P0b.1 Create Shared DialogShell

Problem:

- The modal defect is at the outer panel level.
- Several dialogs have scrollable inner lists, but the panel itself is height-unbounded and the footer sits outside a scrollable region.
- On short viewports, primary buttons can be pushed out of reach.

Model to follow:

- `CreateComparisonDialog.tsx` uses a constrained flex panel and scrollable body.

New file:

- `platform/frontend/src/components/ui/DialogShell.tsx`

Recommended API:

```tsx
interface DialogShellProps {
  isOpen: boolean
  title: React.ReactNode
  description?: React.ReactNode
  icon?: React.ReactNode
  onClose: () => void
  children: React.ReactNode
  footer?: React.ReactNode
  size?: 'sm' | 'md' | 'lg' | 'xl'
  closeDisabled?: boolean
}
```

Required behavior:

- Do not render when `isOpen` is false.
- Backdrop closes when clicked unless `closeDisabled`.
- Escape closes unless `closeDisabled`.
- Panel:

  ```text
  max-h-[calc(100vh-2rem)]
  flex flex-col
  overflow-hidden
  ```

- Header does not scroll.
- Body uses:

  ```text
  flex-1 overflow-y-auto
  ```

- Footer does not scroll and remains reachable.
- Add `role="dialog"` and `aria-modal="true"`.
- Provide accessible close button label.

Acceptance criteria:

- Dialog footer is always reachable at 1366x768, 1024x600, and 390x844.
- Keyboard Escape closes the dialog where allowed.
- Close button has an accessible label.
- Existing dialog visual style remains consistent.

### P0b.2 Refactor Affected Dialogs And Wizards

Refactor in this order:

1. `StartEvaluationWizard`
2. `TestGeneratorWizard`
3. `TestCaseDialog`
4. `CreateProjectDialog`
5. `EditProjectDialog`
6. `ImportTestSetDialog`
7. `CreateTestSetDialog`
8. `CreateIndexDialog`
9. Optional: `CreateKBDialog`
10. Optional: `IndexKBDialog`

Why this order:

- Evaluation and test generation wizards have the highest risk because they contain steppers and footers.
- Test case editing is a common workflow and has large text areas.
- Smaller create dialogs are lower risk but should be standardized.
- Create Index also needs visual design cleanup.

Acceptance criteria:

- Every refactored dialog uses `DialogShell`.
- Header, body, and footer are structurally separated.
- No primary action button can be pushed outside the viewport.
- Existing submit/cancel behavior remains unchanged.
- Existing React Query invalidation behavior remains unchanged.

Manual viewport QA:

- Open every refactored dialog at:
  - Desktop: 1366x768
  - Short desktop: 1024x600
  - Mobile: 390x844
- Confirm:
  - Content scrolls.
  - Footer remains reachable.
  - Backdrop and close button work.
  - Form submission still works.

## Phase P1: Make The Workflow Understandable

### P1.1 Add Project Overview / Setup Checklist

Problem:

- The user has to infer the setup sequence.
- Project Detail opens directly on Knowledge Bases without explaining readiness.

Files likely affected:

- `platform/frontend/src/pages/ProjectDetail.tsx`
- New component under `platform/frontend/src/components/projects/`

Recommended implementation:

1. Add an `Overview` tab as the first project tab.
2. Show project readiness cards or rows:
   - Knowledge bases
   - Documents
   - RAG configs
   - Ready indexes
   - Test sets
   - Evaluations
   - Baseline
3. Show a single next recommended action.
4. Link each readiness item to the correct tab and action.

Example recommended actions:

- "Create a knowledge base."
- "Upload documents."
- "Create a RAG config."
- "Build an index."
- "Create or import a test set."
- "Launch an evaluation."
- "Set a baseline."

Acceptance criteria:

- A new user can understand what is missing without reading documentation.
- Every setup step has a direct action.
- Existing tabs still work.

### P1.2 Add Project Indexes Tab

Problem:

- Indexes are central to the product but not first-class inside Project Detail.
- They are visible under KB Detail and globally, but not as a project workspace tab.

Files likely affected:

- `platform/frontend/src/pages/ProjectDetail.tsx`
- `platform/frontend/src/components/indexes/IndexCard.tsx`
- New component such as `ProjectIndexesTab.tsx`

Recommended implementation:

1. Add an `Indexes` tab between RAG Configs and Evaluations, or between Knowledge Bases and Test Sets.
2. Query indexes with `api.indexes.list({ project_id })`.
3. Show:
   - name
   - status
   - KB
   - RAG config
   - document count
   - chunk count
   - created/build completed time
   - Run Evaluation action for ready indexes
4. Link to Index Detail.

Acceptance criteria:

- Users can see all project indexes in one place.
- Ready indexes can launch evaluations.
- Failed/building/pending indexes have clear status.

### P1.3 Make Project Tabs URL-Backed

Problem:

- Project tabs use local component state.
- URL query params can open a tab, but clicking a tab does not update the URL.

Files:

- `platform/frontend/src/pages/ProjectDetail.tsx`

Recommended implementation:

1. Keep supporting `?tab=...`.
2. When a tab is clicked, call `setSearchParams`.
3. Preserve relevant existing params when appropriate.
4. Default to `overview` or `kb` if no tab param exists.

Acceptance criteria:

- Clicking tabs updates the URL.
- Reloading preserves active tab.
- Browser back/forward navigates tab changes predictably.
- Existing `startEval`, `kbId`, and `indexId` links still work.

### P1.4 Add Routable Detail Pages

Problem:

- Test set detail, evaluation detail, and comparison detail are local state.
- Refreshing or sharing loses context.

Recommended route targets:

```text
/projects/:projectId/test-sets/:testSetId
/projects/:projectId/evaluations/:evaluationId
/projects/:projectId/comparisons/:comparisonId
```

Implementation approach:

1. Start with evaluation detail because it is the highest-value shareable view.
2. Move existing detail components behind route wrappers.
3. Keep existing tab-local behavior temporarily if needed, but redirect users to canonical routes from list cards.
4. Add back links to the owning project tab.

Acceptance criteria:

- Evaluation result pages can be reloaded and shared.
- Test set detail pages can be reloaded and shared.
- Comparison detail pages can be reloaded and shared.
- Back navigation is predictable.

### P1.5 Make Evaluation Launch Context-Aware

Problem:

- The wizard always starts on Test Set.
- If opened from an index, it still asks for KB and Index even though the index determines the KB.

Files:

- `platform/frontend/src/components/evaluations/StartEvaluationWizard.tsx`
- `platform/frontend/src/pages/IndexDetail.tsx`
- `platform/frontend/src/pages/Indexes.tsx`

Recommended implementation:

1. If `initialIndexId` is present, preload its owning KB and index.
2. Consider starting at Test Set but render the selected index as locked context.
3. Do not force users to reselect KB/index when launched from a ready index.
4. Add empty-state actions when prerequisites are missing.

Acceptance criteria:

- Launching from Index Detail opens a wizard that clearly shows the selected index.
- The user only chooses missing inputs.
- The wizard still supports full manual selection when launched from Project Evaluations.

## Phase P2: Improve Operator Efficiency

### P2.1 Improve Search And Filtering

Targets:

- Projects
- Indexes
- Test Sets
- Evaluations
- Playground

Recommended filters:

Projects:

- status
- tag
- readiness

Indexes:

- project
- KB
- status
- RAG type
- stale/current

Test Sets:

- difficulty
- type
- generated/manual
- reviewed/unreviewed

Evaluations:

- status
- test set
- index
- RAG config
- metric threshold
- date

Playground:

- project
- KB
- status
- RAG type

Acceptance criteria:

- Users can narrow long lists without leaving the page.
- Filters are visible, labelled, and resettable.
- Filtering does not require backend changes for the first pass unless data volume requires it.

### P2.2 Show Provenance Everywhere

Problem:

- Evaluation cards and some index surfaces lack enough context for comparison.

Show these fields where available:

- KB
- KB version
- RAG config
- Index
- Test set
- Metrics selected
- Judge model
- Query model
- Created/completed time

Acceptance criteria:

- A user can tell what was evaluated without opening every detail view.
- Evaluation list supports meaningful comparison at a glance.

### P2.3 Add Dense List/Table Modes

Targets:

- Indexes
- Evaluations
- Test cases

Recommended implementation:

1. Keep card views for empty/small states.
2. Add table/list layouts when item counts grow.
3. Make status, score, cost, latency, and provenance sortable where useful.

Acceptance criteria:

- Users can scan many evaluations or indexes without excessive scrolling.
- Important metadata aligns in columns.

### P2.4 Add Stale Index Warnings

Problem:

- Users need to know whether an index reflects the current KB and config state.

Implementation depends on available backend data:

- If index has KB version snapshot, compare it to current KB version.
- If config has updated timestamp and index has config snapshot/build timestamp, show possible staleness.
- If data is insufficient, show only confirmed staleness and avoid guessing.

Acceptance criteria:

- Current indexes are clearly marked.
- Stale or potentially stale indexes are called out before evaluation launch.
- The warning links to rebuild/create index flow.

## Phase P3: Polish, Accessibility, And Scale

### P3.1 Replace Browser Alerts And Confirms

Problem:

- Browser `alert` and `confirm` are used in multiple components.

Recommended implementation:

1. Add shared `ConfirmDialog`.
2. Use it for destructive actions:
   - delete index
   - archive project
   - delete test set
   - delete test case
   - cancel evaluation
   - cancel generation
3. Replace alert-based errors with toasts or inline error banners.

Acceptance criteria:

- No routine destructive action uses browser confirm.
- User-facing errors are styled consistently.

### P3.2 Add Tooltip And Accessibility Labels

Targets:

- icon-only edit/delete/export/run buttons
- close buttons
- collapsed/hover-only actions

Acceptance criteria:

- Icon-only buttons have `aria-label`.
- Tooltips explain non-obvious actions.
- Keyboard focus states are visible.

### P3.3 Add Mobile Navigation

Problem:

- Current layout uses fixed desktop sidebar.

Recommended implementation:

1. Add responsive sidebar behavior.
2. On mobile:
   - hide sidebar behind menu button, or
   - use bottom/top navigation for primary pages.
3. Ensure main content has usable horizontal constraints.

Acceptance criteria:

- App is navigable at common mobile widths.
- Sidebar does not compress main content unusably.
- Dialogs remain usable on mobile after P0b.

### P3.4 Code-Split Route Bundles

Problem:

- Production build warns that the main bundle is large.

Recommended implementation:

1. Lazy-load route pages with `React.lazy`.
2. Consider splitting heavy result/comparison/chart components.
3. Keep loading states consistent.

Acceptance criteria:

- `npm.cmd run build` no longer emits the large chunk warning, or the warning is substantially reduced.
- Route transitions remain stable.

## Suggested Work Order

1. P0a.1 Sidebar route cleanup.
2. P0a.2 Sidebar positioning.
3. P0a.3 RAG config broken link.
4. P0a.4 Archive button and className bug.
5. P0a.5 Global Indexes search and Run Evaluation.
6. P0a.6 Playground naming.
7. P0b.1 Build `DialogShell`.
8. P0b.2 Refactor dialogs in priority order.
9. P1.1 Project Overview/setup checklist.
10. P1.2 Project Indexes tab.
11. P1.3 URL-backed tabs.
12. P1.4 Routable detail pages.
13. P1.5 Context-aware evaluation launch.
14. P2 and P3 improvements.

## Testing And Verification Plan

Run from `platform/frontend`:

```powershell
npm.cmd run lint
npm.cmd run build
```

Manual QA after P0a:

- Click every sidebar item.
- Open Index Detail and click the RAG config reference.
- Archive a project and verify status/query refresh.
- Use global Indexes search.
- Launch evaluation from a ready index on global Indexes.
- Confirm Playground heading uses index terminology.

Manual QA after P0b:

- Open every refactored dialog.
- Test at:
  - 1366x768
  - 1024x600
  - 390x844
- Confirm footer buttons are reachable.
- Confirm body scroll works.
- Confirm Escape and close button work.
- Confirm submit/cancel still behave correctly.

Manual QA after P1:

- Share/reload URLs for:
  - Project tab
  - Test set detail
  - Evaluation detail
  - Comparison detail
- Follow setup checklist from a new empty project to first evaluation.
- Launch evaluation from:
  - Project Evaluations tab
  - Index Detail
  - Global Indexes page

## Definition Of Done

The UI remediation is complete when:

- No visible navigation link opens 404.
- No primary modal action is unreachable on short viewports.
- Every visible action either works or is removed.
- The project workflow clearly explains the dependency chain from KB/config to index to evaluation.
- Evaluation and comparison detail views are shareable/reloadable.
- Indexes are visible and actionable inside the project context.
- All frontend checks pass.
