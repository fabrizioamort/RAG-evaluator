# UI Review

Date: 2026-05-31

Scope: `platform/frontend`

This review covers the current React UI structure, navigation, workflow model, entity relationships, known functional defects, and missing UX pieces needed for a complete RAG evaluation product.

Build checks run during review:

- `npm.cmd run build`: passed
- `npm.cmd run lint`: passed

The build emits a Vite chunk-size warning because the main JavaScript bundle is large. This is not blocking, but it should be addressed later with route-level code splitting.

Verification note: the items in "Critical Bugs And Functional Defects" were confirmed directly against the source and carry `file:line` references. The workflow and information-architecture observations further down are a mix of confirmed behavior and reasoned inference from the code; where wording is tentative ("likely," "can," "may"), treat it as inferred and worth reproducing before acting.

## Executive Summary

The UI has the correct core entities for a RAG evaluation platform:

- Projects
- Knowledge bases
- Documents
- RAG configurations
- Indexes
- Test sets
- Evaluations
- Comparisons
- Trends
- Playground queries

The main product model should be project-centered:

1. A project groups the evaluation work.
2. A knowledge base contains source documents.
3. A RAG configuration defines retrieval and generation behavior.
4. A knowledge base plus a RAG configuration produces an index.
5. A test set defines questions and expected answers.
6. An evaluation runs one index against one test set using selected metrics and runtime overrides.
7. Comparisons and trends are derived from completed evaluations.
8. The playground lets users query one or more ready indexes interactively.

That model exists in the code, but the UI does not make it explicit enough. The user must infer the required setup sequence, and several navigation elements imply global pages that do not actually exist. The biggest immediate problems are broken links, non-scrollable long dialogs, incomplete actions, and inconsistent page styling.

Overall, the UI is a promising operational dashboard, but it needs a stronger workflow spine and tighter route model before it will feel complete and practical for daily use.

## Critical Bugs And Functional Defects

### 1. Sidebar Links Point To Missing Routes

The sidebar includes these links:

- `/evaluations`
- `/trends`
- `/settings`

Only these routes are currently registered:

- `/`
- `/projects`
- `/projects/:id`
- `/knowledge-bases/:id`
- `/indexes`
- `/indexes/:id`
- `/playground`

Result: clicking Evaluations, Trends, or Settings opens the 404 page.

Relevant files:

- `platform/frontend/src/components/layout/Layout.tsx` (nav at `:18-19`, Settings at `:52`)
- `platform/frontend/src/App.tsx` (registered routes at `:14-23`)

Recommended fix:

- Remove the sidebar links until real global pages exist, or implement global pages for evaluations and trends.
- Settings should either be implemented or removed.
- If evaluations and trends remain project-scoped, they should be reached from Project Detail, not from global sidebar links.

### 2. RAG Config Link Is Broken From Index Detail

Index Detail links to `/rag-configs/:id`, but there is no route for that path.

Relevant file:

- `platform/frontend/src/pages/IndexDetail.tsx:196`

Recommended fix:

- Either add a `RAGConfigDetail` route, or change the link to navigate back to the owning project with the RAG Configs tab open.
- Better long term: add a real route such as `/projects/:projectId/rag-configs/:configId`.

### 3. Long Modals And Wizards Can Become Unusable On Small Viewports

The shared defect is at the **outer modal panel**: it is height-unbounded, and the footer holding the primary action buttons sits outside any scroll region. On shorter screens the panel grows taller than the viewport and pushes buttons like Continue, Generate, Save, or Start out of reach.

This is specifically the outer panel, not the inner content. Two of the wizards already scroll their inner selection lists (`StartEvaluationWizard.tsx:298,327,367` and `TestGeneratorWizard.tsx:237,391` use `max-h-[300px]/[280px] overflow-y-auto`), so a reader who greps the files will find `overflow-y-auto` and may wrongly conclude the dialog is fine. The defect is one level up: the panel itself (`StartEvaluationWizard.tsx:248`, `relative w-full max-w-3xl` with no `max-h` or `overflow`) is unbounded and its footer is outside the scroll region. The fix is to constrain and scroll the panel, not the lists.

Affected dialogs and wizards (outer panel unbounded):

- `StartEvaluationWizard`
- `TestGeneratorWizard`
- `TestCaseDialog`
- `CreateProjectDialog`
- `EditProjectDialog`
- `ImportTestSetDialog`
- `CreateTestSetDialog`
- `CreateIndexDialog`

One dialog already solves this at the panel level and is the model to standardize on: `CreateComparisonDialog.tsx:60` uses `flex max-h-[85vh] ... flex-col` on the panel with a scrollable body at `:71` (`flex-1 overflow-y-auto`).

Recommended fix:

- Create a shared `DialogShell` component with:
  - `fixed inset-0`
  - centered panel
  - `max-h-[calc(100vh-2rem)]`
  - `flex flex-col`
  - non-scrolling header
  - `overflow-y-auto` body
  - sticky or non-scrolling footer
  - Escape-to-close support
  - focus trap
  - proper ARIA roles
- Refactor all dialogs and wizards to use it.

### 4. Sidebar Settings Positioning Is Fragile

The settings link is absolutely positioned (`absolute bottom-4 left-4 right-4`, `Layout.tsx:50`) inside an `aside` (`Layout.tsx:26`) that has no `relative`. This can cause the settings link to anchor relative to the wrong container and potentially overlay main UI.

Relevant file:

- `platform/frontend/src/components/layout/Layout.tsx:26,50`

Recommended fix:

- Make the sidebar `relative flex flex-col`.
- Put main nav in a `flex-1` region.
- Put Settings as a normal footer item.

### 5. Archive Project Button Does Nothing

Project Detail shows an `Archive` button (`ProjectDetail.tsx:726`), but it has no `onClick` handler.

The same button also has a CSS bug: its className uses backslashes instead of Tailwind opacity slashes (`bg-destructive\10`, `border-destructive\20` instead of `bg-destructive/10`, `border-destructive/20`). Those are invalid utilities, so the intended background and border tint are not applied today.

The endpoint already exists: `api.projects.archive` is defined at `client.ts:703` (`POST /projects/{id}/archive`), so wiring is straightforward.

Relevant file:

- `platform/frontend/src/pages/ProjectDetail.tsx:726`

Recommended fix:

- Wire it to `api.projects.archive` (already available, `client.ts:703`).
- Fix the backslash class names (`/10`, `/20`).
- Add a confirmation dialog.
- Invalidate project and project list queries.
- Update UI status after archiving.

### 6. Global Indexes Page Has Incomplete Controls

The Indexes page includes:

- A disabled search input with a no-op `onChange` (`Indexes.tsx:54-63`).
- A ready-index Run Evaluation icon that only logs to console (`Indexes.tsx:101`).

Relevant file:

- `platform/frontend/src/pages/Indexes.tsx:54-63,101`

Recommended fix:

- Either remove disabled search or implement client-side filtering while backend search is unavailable.
- Wire Run Evaluation to the same project evaluation launch path used by Index Detail.
- Make the action visible and labelled enough to be discoverable.

### 7. Hover-Only Actions Hide Important Controls

Several delete, edit, export, and menu actions are hidden with `opacity-0 group-hover:opacity-100`.

This is weak for:

- Touch devices.
- Keyboard users.
- Discoverability.
- Dense admin workflows where users need to scan available actions.

Recommended fix:

- Keep primary actions visible.
- Keep destructive actions either visible in a compact menu or behind a clearly visible actions button.
- Add accessible labels and tooltips for icon-only buttons.

## Information Architecture Review

### Current Structure

The current top-level navigation is:

- Dashboard
- Projects
- Indexes
- Playground
- Evaluations
- Trends
- Settings

The actual implemented product structure is:

- Dashboard
- Projects
- Project Detail tabs:
  - Knowledge Bases
  - Test Sets
  - RAG Configs
  - Evaluations
  - Comparisons
  - Trends
- Knowledge Base Detail
- Global Indexes
- Index Detail
- Playground

This creates a mismatch. Evaluations and Trends are implemented as project-scoped tabs, while the sidebar presents them as global sections. RAG Configs are project-scoped, but Index Detail links to them as if they had a global page.

### Recommended Product Model

The most coherent structure is:

- Dashboard
- Projects
- Indexes
- Playground

Inside each project:

- Overview
- Knowledge Bases
- RAG Configs
- Indexes
- Test Sets
- Evaluations
- Comparisons
- Trends

This would make the main workflow clear:

1. Create or open a project.
2. Add a knowledge base.
3. Upload documents.
4. Create a RAG configuration.
5. Build an index from the KB and config.
6. Create, import, or generate a test set.
7. Launch an evaluation using a test set and ready index.
8. Review results.
9. Compare evaluations.
10. Track trends over time.

### Missing Project Overview

Project Detail currently starts on the Knowledge Bases tab. That is reasonable for setup, but it does not provide an overview of project readiness.

Recommended addition:

- Add a project Overview tab or summary band showing:
  - KB count and document count
  - RAG config count
  - Ready index count
  - Test set count
  - Completed/running/failed evaluation count
  - Current baseline evaluation
  - Next recommended action

Example states:

- "Upload documents to a knowledge base."
- "Create a RAG config before building an index."
- "Build an index to make this project evaluable."
- "Create or import a test set."
- "Ready to run an evaluation."
- "Set a baseline to compare future runs."

## Workflow Review

### Dashboard

Strengths:

- Gives useful high-level counts.
- Shows API health.
- Shows recent activity.
- Provides quick actions.

Weaknesses:

- Quick actions are too generic. Upload Documents, Run Evaluation, and Generate Test Set all navigate to `/projects` instead of starting the requested workflow.
- Recent activity only navigates projects and knowledge bases. Evaluation and test set activity is visible but not actionable.
- The dashboard does not explain project readiness or outstanding setup tasks.

Recommended fixes:

- Make each quick action either:
  - Open a project picker and continue into the correct workflow.
  - Navigate to the most recent active project and open the right tab/dialog.
- Recent activity should deep-link to:
  - Evaluation result/progress
  - Test set detail
  - KB detail
  - Project detail
- Add a "Continue setup" or "Ready projects" area.

### Projects

Strengths:

- Project cards communicate counts for KBs, test sets, RAG configs, and evaluations.
- Empty state is clear.
- Create project dialog is straightforward.

Weaknesses:

- There is no search, filter, or status grouping.
- Project cards do not show ready/evaluable status.
- The More button on project cards is inert.

Recommended fixes:

- Add project search and status filter.
- Add readiness indicators:
  - No KB
  - Has documents
  - Has ready index
  - Has test set
  - Has completed evaluations
- Wire or remove the More button.

### Project Detail

Strengths:

- Project tabs mostly match the domain.
- Editing project metadata exists.
- Evaluation, comparison, and trend functionality live near the project context.

Weaknesses:

- No project Overview tab.
- Tabs are local state. Clicking a tab does not update the URL, so users cannot share or restore tab state.
- Detail views for test sets, evaluations, and comparisons are local state, not real routes.
- Archive button is not functional.
- The relationship between KBs, RAG Configs, Indexes, and Evaluations is not explained.

Recommended fixes:

- Add URL-backed tabs, for example `?tab=evals` when switching tabs.
- Add project-scoped nested routes for detail views.
- Add a setup checklist.
- Add an Indexes tab inside Project Detail.
- Implement Archive.

### Knowledge Bases

Strengths:

- KB cards show document count, version, and status.
- KB detail supports document upload and deletion.
- KB detail shows indexes.
- Create Index action is available from KB card and KB detail.

Weaknesses:

- Create Index assumes the user understands that a RAG Config is required.
- If no RAG configs exist, the dialog likely feels like a dead end.
- KB document processing status is underdeveloped. There is no clear reprocessing or failure recovery workflow.
- There is no clear stale-index warning when documents change after an index was built.

Recommended fixes:

- In Create Index, if no RAG configs exist, show a clear empty state with "Create RAG Config".
- Show index freshness relative to KB version.
- Add document upload status progress and failed-file details.
- Add a warning when deleting documents may invalidate indexes.

### RAG Configs

Strengths:

- Config dialog supports RAG type, provider, model, embedding model, and typed parameters.
- Build-time and query-time parameters are separated in the dialog, which is conceptually correct.

Weaknesses:

- No dedicated detail page or expanded view.
- Config cards show too little of the settings.
- The relationship "configs are templates; indexes freeze build-time settings" is not strongly communicated.
- Naming is inconsistent, though in exactly one place: the Playground heading says "Select RAG Systems to Compare" (`Playground.tsx:165`) while every other surface says RAG Config. A one-line fix.

Recommended fixes:

- Add a config detail/preview view.
- Show which indexes were built from each config.
- Show build-time vs query-default parameters directly in the card or detail view.
- Use consistent language:
  - RAG Config: reusable configuration.
  - Index: built artifact from a KB plus RAG Config.
  - Evaluation: run of an Index against a Test Set.

### Indexes

Strengths:

- Index Detail has useful build metadata and config snapshot.
- Index Detail has a Run Evaluation path.
- KB Detail lists indexes in context.

Weaknesses:

- Global Indexes page uses inconsistent styling.
- Search is disabled.
- Run Evaluation from global Indexes does not work.
- Indexes are not present as a first-class project tab.
- Index Detail links to a missing RAG Config route.
- Index cards do not clearly show KB version, build duration, build error details, or last-used evaluation context.

Recommended fixes:

- Add Project Detail > Indexes tab.
- Make global Indexes a cross-project inventory, not the primary workflow.
- Add filters:
  - Project
  - KB
  - RAG type
  - Status
  - Stale/current
- Wire Run Evaluation everywhere a ready index appears.
- Replace the missing RAG Config link with a valid project-scoped destination.

### Test Sets

Strengths:

- Supports empty set creation, JSON import, manual cases, generation, review, export.
- Test case table has useful metadata and search.
- Generation workflow is conceptually strong.

Weaknesses:

- Test set detail is not routable.
- Create/edit test case dialog can overflow viewport.
- Test case status labels are not fully clear. Manual cases can appear "Pending" even when review may only matter for generated cases.
- Export in list calls an empty handler in Project Detail list context.

Recommended fixes:

- Add real route for test set detail.
- Make test case dialog scrollable.
- Separate generated review status from manual case status.
- Implement export from both list and detail.
- Add filters for difficulty, type, generated/manual, reviewed/unreviewed.

### Evaluation Launch

Strengths:

- The wizard selects test set, KB, index, query overrides, metrics, and review.
- It distinguishes frozen build parameters from query-time overrides.
- It supports initial KB/index from Index Detail, which is useful.

Weaknesses:

- Wizard can overflow viewport.
- It starts on Test Set even when opened from an index. If an index is preselected, the flow should adapt.
- It asks for both KB and Index even though the index already determines the KB.
- Empty states are weak when there are no test sets, no KBs, or no ready indexes.
- There is no cost/time estimate before launch.

Recommended fixes:

- Make the wizard route/context aware.
- If started from an index, skip or prefill KB and Index.
- Consider selecting Index before KB, because for evaluation the executable artifact is the index.
- Add prerequisite empty states with direct actions:
  - Create Test Set
  - Upload Documents
  - Create RAG Config
  - Build Index
- Add estimated cost and runtime based on test case count and selected metrics.

### Evaluation Results

Strengths:

- Rich result detail.
- Metric summaries are useful.
- Per-result expansion is practical.
- Manifest tab is an important feature.
- Baseline setting exists.

Weaknesses:

- Result view is not routable.
- Evaluation list cards do not show enough provenance.
- Search requires Enter, which is easy to miss.
- There is no filtering by failed/low-score cases, difficulty, category, metric threshold, or latency/cost.
- Baseline setting is an inline panel rather than a focused confirmation dialog.

Recommended fixes:

- Add routes:
  - `/projects/:projectId/evaluations/:evaluationId`
  - `/projects/:projectId/evaluations/:evaluationId/results/:resultId`
- Add filters:
  - Metric below threshold
  - Difficulty
  - Category
  - Has retrieval trace
  - Failed/error results
- Add sortable columns or a dense table mode.
- Show provenance on evaluation list:
  - Test set
  - Index
  - KB
  - RAG config
  - Judge model
  - Metrics selected

### Comparisons

Strengths:

- Comparison creation supports multiple evaluations.
- Warns about mixed test sets.
- Provides metrics, charts, per-question, and config-diff sections.
- Baseline can be switched within detail.

Weaknesses:

- Comparison detail is not routable.
- Create comparison assumes first selected evaluation is baseline, which is fragile.
- There is no strong guidance about what makes evaluations comparable.

Recommended fixes:

- Add a routable comparison detail page.
- Let users explicitly choose baseline.
- Group evaluations by test set and KB/index family in the selector.
- Disable or warn more prominently when test sets differ.

### Trends

Strengths:

- Project-scoped trends are the right location.
- Metric trends and efficiency maps are useful concepts.

Weaknesses:

- Sidebar implies global Trends, but only project trends exist.
- Trends are only useful after multiple completed evaluations, but the empty/prerequisite state is not emphasized.

Recommended fixes:

- Remove global Trends sidebar link or implement a global trends page.
- In project trends, show clear empty states:
  - "Run at least two completed evaluations to see trends."
  - "Set a baseline to interpret changes."

### Playground

Strengths:

- Lets users compare up to four indexes interactively.
- Groups indexes by KB.
- Displays answer, retrieved chunks, trace, latency, tokens, and cost.
- Query history exists.

Weaknesses:

- Playground is global but lacks project/KB filters in the UI.
- Selected indexes from different projects or KBs may be compared without enough context.
- Query controls can become cramped on smaller screens.
- Results for three or four indexes still use a two-column grid, which can create long pages and difficult comparison.

Recommended fixes:

- Add project and KB filters.
- Add a comparison table mode for multiple results.
- Make selected index context clearer: project, KB, RAG config, index status.
- Make query settings responsive.

## Visual Design And Interaction Quality

### Strengths

- The core style is calm and work-focused, which fits a RAG evaluation platform.
- The project-level screens mostly use restrained colors and readable spacing.
- Icons help scanning when used consistently.
- Evaluation Results and Comparisons have good information density.

### Weaknesses

- Styling is inconsistent between newer project components and older index components.
- Some pages use raw gray/blue Tailwind styling instead of design tokens.
- Cards are overused for operational data that would sometimes work better as tables or dense lists.
- Several icon-only controls lack labels or tooltips.
- Browser `alert` and `confirm` are used in many places, which feels unfinished and inconsistent.
- Some copy is generic instead of task-oriented.

Recommended fixes:

- Standardize all pages on the same token system:
  - `bg-card`
  - `border-border`
  - `text-muted-foreground`
  - `text-primary`
- Replace browser confirms with shared confirmation dialogs.
- Use tables/lists for evaluations, indexes, and test cases where scanning matters.
- Add tooltips and accessible labels to icon-only buttons.
- Keep action buttons visible on touch-oriented surfaces.

## Accessibility Review

Important gaps:

- Dialogs do not implement focus trap.
- Dialogs do not consistently support Escape close.
- Dialogs lack explicit `role="dialog"` and `aria-modal`.
- Icon-only buttons often lack `aria-label`.
- Hover-only controls are difficult for keyboard and touch users.
- Sidebar is not responsive for mobile.
- Some click targets are cards without semantic button/link roles.

Recommended fixes:

- Introduce shared Dialog, ConfirmDialog, and Tooltip primitives.
- Use actual `button` or `Link` elements for interactive cards, or apply correct roles and keyboard handlers.
- Add visible focus states.
- Add mobile navigation.

## Data State And Error Handling

Current strengths:

- Loading states exist in many screens.
- Toasts are used for several mutations.
- React Query is used consistently for server data.

Current weaknesses:

- Some errors are only logged to console.
- Some mutation failures use browser alerts.
- Some empty states do not provide next actions.
- Invalidation is inconsistent. For example, creating an index from a KB card invalidates `indexes`, but KB-specific index fetches are local state in some places.

Recommended fixes:

- Standardize mutation handling:
  - Toast success
  - Toast error with backend detail
  - Query invalidation by entity and project
- Prefer React Query for indexes everywhere instead of local fetch state.
- Add actionable empty/error states.

## Suggested Route Model

Recommended route structure:

```text
/
/projects
/projects/:projectId
/projects/:projectId/knowledge-bases
/projects/:projectId/knowledge-bases/:kbId
/projects/:projectId/rag-configs
/projects/:projectId/rag-configs/:configId
/projects/:projectId/indexes
/projects/:projectId/indexes/:indexId
/projects/:projectId/test-sets
/projects/:projectId/test-sets/:testSetId
/projects/:projectId/evaluations
/projects/:projectId/evaluations/:evaluationId
/projects/:projectId/comparisons
/projects/:projectId/comparisons/:comparisonId
/projects/:projectId/trends
/indexes
/playground
/settings
```

Top-level `/indexes` should be an inventory/search page. The main creation and evaluation workflow should remain project-scoped.

Note that this model defines two paths to the same index detail: `/projects/:projectId/indexes/:indexId` and the existing `/indexes/:id`. Pick one as canonical (the project-scoped path is the better default) and have the other redirect to it, so the two routes do not render the detail view inconsistently.

## Prioritized Remediation Plan

### P0a: Quick Wins (small wiring fixes, roughly an hour each)

1. Remove or implement missing sidebar routes.
2. Fix `/rag-configs/:id` broken link.
3. Fix sidebar Settings positioning.
4. Wire or remove Project Archive (and fix its backslash class names).
5. Wire or remove incomplete global Indexes actions.

### P0b: Shared Dialog Refactor (a real component plus refactor of 8 dialogs)

1. Build a `DialogShell` that constrains and scrolls the outer panel, modeled on `CreateComparisonDialog`.
2. Refactor the affected dialogs and wizards to use it.

This is a larger, separate effort and should not block the P0a quick wins.

### P1: Make The Workflow Understandable

1. Add Project Overview or setup checklist.
2. Add Project Indexes tab.
3. Add URL-backed project tabs.
4. Add routable detail pages for test sets, evaluations, and comparisons.
5. Make Evaluation Launch adapt to context.
6. Improve empty states for missing prerequisites.

### P2: Improve Daily Operator Efficiency

1. Add filters and search to Projects, Indexes, Test Sets, Evaluations, and Playground.
2. Add dense table/list modes for indexes and evaluations.
3. Show provenance everywhere:
   - KB
   - KB version
   - RAG config
   - Index
   - Test set
   - Metrics
   - Judge model
4. Add stale-index warnings.
5. Add evaluation cost/runtime estimates.

### P3: Polish And Scale

1. Replace browser alerts/confirms with app dialogs.
2. Add tooltips and ARIA labels.
3. Add mobile navigation.
4. Code-split route bundles.
5. Standardize legacy-styled pages with the app design system.

## Final Assessment

The current UI is close to being a useful RAG evaluation workbench, but it needs a clearer product spine. The system should teach users the core dependency chain:

```text
Project
  -> Knowledge Base + RAG Config
  -> Index
  -> Test Set
  -> Evaluation
  -> Results, Comparison, Trends
```

Right now that relationship exists in the data model and API, but the UI exposes it unevenly. Once the broken links, modal scrolling, and incomplete actions are fixed, the next highest-impact improvement is a project-level setup/overview experience that shows what exists, what is missing, and what action the user should take next.
