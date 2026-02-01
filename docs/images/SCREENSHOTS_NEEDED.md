# Screenshots & Images Needed

> **Complete list of screenshots and images needed for the RAG Evaluator Platform documentation.**

This document lists all the visual assets required to complete the documentation. Each entry includes:

- **Filename**: Where to save the image
- **Description**: What the image should show
- **Dimensions**: Recommended size
- **Priority**: High, Medium, or Low

---

## How to Capture Screenshots

### Recommended Tools

- **macOS**: Built-in Screenshot (Cmd+Shift+4) or CleanShot X
- **Windows**: Snipping Tool (Win+Shift+S) or ShareX
- **Linux**: Flameshot or GNOME Screenshot

### Best Practices

1. Use a clean browser profile (no extensions visible)
2. Use light mode for consistency
3. Ensure no sensitive data is visible (API keys, passwords)
4. Use realistic but fake data for examples
5. Save as PNG for UI screenshots, SVG for diagrams

---

## Priority: HIGH (Required for README & Getting Started)

### README Assets

| # | Filename | Description | Dimensions |
|---|----------|-------------|------------|
| 1 | `logo.png` | Platform logo (icon style, transparent background) | 120x120px |
| 2 | `hero-screenshot.png` | Main dashboard showing projects list with sample data | 1600x900px |

### Getting Started Guide

| # | Filename | Description | Dimensions |
|---|----------|-------------|------------|
| 3 | `getting-started-dashboard.png` | Initial dashboard  | 1200x700px |
| 4 | `getting-started-create-project.png` | Create project dialog with fields filled | 600x400px |
| 5 | `getting-started-upload-docs.png` | Document upload interface with drag-drop zone active | 800x500px |
| 6 | `getting-started-build-index.png` | Index building progress bar at ~50% | 800x400px |
| 7 | `getting-started-test-set.png` | Test set creation with sample questions | 1000x600px |
| 8 | `getting-started-start-eval.png` | Evaluation wizard with all options selected | 700x500px |
| 9 | `getting-started-eval-progress.png` | Evaluation in progress showing real-time updates | 1000x600px |
| 10 | `getting-started-results-summary.png` | Completed evaluation summary with all metric cards | 1200x700px |
| 11 | `getting-started-per-case.png` | Expanded per-case result showing all details | 1000x600px |
| 12 | `getting-started-explainability.png` | Metric explainability panel open | 600x400px |
| 13 | `getting-started-retrieval-trace.png` | Retrieval trace viewer showing chunks | 1000x600px |
| 14 | `getting-started-comparison.png` | Side-by-side comparison of two evaluations | 1200x600px |
| 15 | `getting-started-trends.png` | Trends dashboard with line chart | 1200x600px |

---

## Priority: MEDIUM (Architecture & Evaluation Guide)

### Architecture Documentation

| # | Filename | Description | Dimensions |
|---|----------|-------------|------------|
| 16 | `architecture-overview.png` | High-level 3-tier architecture diagram | 1200x800px |
| 17 | `core-engine-architecture.png` | Detailed core engine component diagram | 1000x700px |
| 18 | `backend-architecture.png` | Backend service layer diagram | 1000x600px |
| 19 | `frontend-architecture.png` | React component hierarchy diagram | 1000x700px |
| 20 | `database-schema.png` | Entity-relationship diagram | 1200x900px |
| 21 | `vector-stores.png` | Vector store comparison diagram | 800x500px |
| 22 | `document-ingestion-flow.png` | Sequence diagram for document upload | 1000x600px |
| 23 | `evaluation-flow.png` | Sequence diagram for evaluation | 1000x700px |
| 24 | `evaluation-pipeline.png` | Evaluation pipeline diagram | 1200x700px |

### Evaluation Guide

| # | Filename | Description | Dimensions |
|---|----------|-------------|------------|
| 25 | `eval-guide-test-composition.png` | Pie chart of ideal test set composition | 600x400px |
| 26 | `eval-guide-test-generation.png` | AI test generation wizard | 800x500px |
| 27 | `eval-guide-wizard.png` | Full start evaluation wizard | 700x600px |
| 28 | `eval-guide-choose-metrics.png` | Metric selection checkboxes | 500x400px |
| 29 | `eval-guide-review-run.png` | Review before starting evaluation | 600x400px |
| 30 | `eval-guide-progress.png` | Real-time evaluation progress | 1000x600px |
| 31 | `eval-guide-results-summary.png` | Complete results with all metrics | 1200x800px |
| 32 | `eval-guide-per-case.png` | Expanded single test case result | 1000x600px |
| 33 | `eval-guide-explainability.png` | Detailed metric explanation | 700x500px |
| 34 | `eval-guide-trace.png` | Full retrieval trace view | 1000x700px |
| 35 | `eval-guide-comparison.png` | Multi-evaluation comparison table | 1200x600px |
| 36 | `eval-guide-baseline.png` | Baseline comparison view | 1000x500px |
| 37 | `eval-guide-trends.png` | Trends dashboard with filters | 1200x700px |

---

## Priority: MEDIUM (UI Guide)

### UI Guide Screenshots

| # | Filename | Description | Dimensions |
|---|----------|-------------|------------|
| 38 | `ui-dashboard-overview.png` | Full dashboard with projects and stats | 1400x800px |
| 39 | `ui-create-project.png` | Create project dialog | 600x400px |
| 40 | `ui-project-detail.png` | Full project detail page | 1400x900px |
| 41 | `ui-kb-section.png` | Knowledge bases list section | 1000x400px |
| 42 | `ui-testsets-section.png` | Test sets list section | 1000x400px |
| 43 | `ui-ragconfigs-section.png` | RAG configs list with badges | 1000x400px |
| 44 | `ui-evaluations-section.png` | Evaluations list with statuses | 1000x400px |
| 45 | `ui-edit-project.png` | Edit project dialog | 600x400px |
| 46 | `ui-create-kb.png` | Create knowledge base dialog | 500x350px |
| 47 | `ui-upload-docs.png` | Document upload with files listed | 800x500px |
| 48 | `ui-document-list.png` | Document list with metadata | 800x400px |
| 49 | `ui-build-index.png` | Index building wizard | 600x500px |
| 50 | `ui-index-progress.png` | Index building progress | 600x300px |
| 51 | `ui-index-detail.png` | Completed index details | 800x400px |
| 52 | `ui-create-testset.png` | Create test set dialog | 500x350px |
| 53 | `ui-add-testcase.png` | Add test case form | 600x500px |
| 54 | `ui-import-testcases.png` | JSON import dialog | 600x400px |
| 55 | `ui-generate-testcases.png` | AI generation wizard | 700x500px |
| 56 | `ui-review-generated.png` | Review generated cases | 1000x600px |
| 57 | `ui-create-ragconfig.png` | Create RAG config wizard | 700x600px |
| 58 | `ui-llm-settings.png` | LLM provider/model selection | 600x400px |
| 59 | `ui-vector-params.png` | Vector semantic parameters | 500x300px |
| 60 | `ui-hybrid-params.png` | Hybrid search parameters | 500x300px |
| 61 | `ui-graph-params.png` | Graph RAG parameters | 500x350px |
| 62 | `ui-filesystem-params.png` | Filesystem RAG parameters | 500x350px |
| 63 | `ui-start-evaluation.png` | Start evaluation full wizard | 800x600px |
| 64 | `ui-choose-metrics.png` | Metric selection UI | 500x400px |
| 65 | `ui-review-run.png` | Pre-evaluation review | 600x400px |
| 66 | `ui-eval-progress.png` | Live evaluation progress | 1000x600px |
| 67 | `ui-results-summary.png` | Full results summary | 1200x800px |
| 68 | `ui-metric-cards.png` | Close-up of metric cards | 800x200px |
| 69 | `ui-per-case-results.png` | Results table with expansion | 1200x600px |
| 70 | `ui-explainability.png` | Explainability panel open | 600x500px |
| 71 | `ui-retrieval-trace.png` | Full retrieval trace | 1000x700px |
| 72 | `ui-export-results.png` | Export dropdown menu | 300x200px |
| 73 | `ui-comparison.png` | Side-by-side comparison | 1200x600px |
| 74 | `ui-set-baseline.png` | Set baseline interface | 400x200px |
| 75 | `ui-trends-dashboard.png` | Full trends view | 1400x800px |
| 76 | `ui-trend-filters.png` | Trend filter options | 400x300px |

---

## Priority: LOW (Security & Advanced)

### Security Guide

| # | Filename | Description | Dimensions |
|---|----------|-------------|------------|
| 77 | `security-network-architecture.png` | Network security diagram with zones | 1000x700px |

---

## Diagram Creation Tools

For technical diagrams, consider using:

- **[Excalidraw](https://excalidraw.com/)** - Hand-drawn style diagrams
- **[draw.io](https://app.diagrams.net/)** - Professional diagrams
- **[Mermaid Live Editor](https://mermaid.live/)** - Code-based diagrams
- **[Lucidchart](https://www.lucidchart.com/)** - Enterprise diagrams

---

## File Organization

Save all images to:

```
docs/images/
├── logo.png
├── hero-screenshot.png
├── getting-started-*.png
├── architecture-*.png
├── eval-guide-*.png
├── ui-*.png
└── security-*.png
```

---

## Checklist

Use this checklist to track progress:

### High Priority (17 images)

- [ ] logo.png
- [ ] hero-screenshot.png
- [ ] getting-started-dashboard.png
- [ ] getting-started-create-project.png
- [ ] getting-started-upload-docs.png
- [ ] getting-started-build-index.png
- [ ] getting-started-test-set.png
- [ ] getting-started-start-eval.png
- [ ] getting-started-eval-progress.png
- [ ] getting-started-results-summary.png
- [ ] getting-started-per-case.png
- [ ] getting-started-explainability.png
- [ ] getting-started-retrieval-trace.png
- [ ] getting-started-comparison.png
- [ ] getting-started-trends.png

### Medium Priority - Architecture (9 images)

- [ ] architecture-overview.png
- [ ] core-engine-architecture.png
- [ ] backend-architecture.png
- [ ] frontend-architecture.png
- [ ] database-schema.png
- [ ] vector-stores.png
- [ ] document-ingestion-flow.png
- [ ] evaluation-flow.png
- [ ] evaluation-pipeline.png

### Medium Priority - Eval Guide (13 images)

- [ ] eval-guide-test-composition.png
- [ ] eval-guide-test-generation.png
- [ ] eval-guide-wizard.png
- [ ] eval-guide-choose-metrics.png
- [ ] eval-guide-review-run.png
- [ ] eval-guide-progress.png
- [ ] eval-guide-results-summary.png
- [ ] eval-guide-per-case.png
- [ ] eval-guide-explainability.png
- [ ] eval-guide-trace.png
- [ ] eval-guide-comparison.png
- [ ] eval-guide-baseline.png
- [ ] eval-guide-trends.png

### Medium Priority - UI Guide (39 images)

- [ ] ui-dashboard-overview.png
- [ ] ui-create-project.png
- [ ] ui-project-detail.png
- [ ] ui-kb-section.png
- [ ] ui-testsets-section.png
- [ ] ui-ragconfigs-section.png
- [ ] ui-evaluations-section.png
- [ ] ui-edit-project.png
- [ ] ui-create-kb.png
- [ ] ui-upload-docs.png
- [ ] ui-document-list.png
- [ ] ui-build-index.png
- [ ] ui-index-progress.png
- [ ] ui-index-detail.png
- [ ] ui-create-testset.png
- [ ] ui-add-testcase.png
- [ ] ui-import-testcases.png
- [ ] ui-generate-testcases.png
- [ ] ui-review-generated.png
- [ ] ui-create-ragconfig.png
- [ ] ui-llm-settings.png
- [ ] ui-vector-params.png
- [ ] ui-hybrid-params.png
- [ ] ui-graph-params.png
- [ ] ui-filesystem-params.png
- [ ] ui-start-evaluation.png
- [ ] ui-choose-metrics.png
- [ ] ui-review-run.png
- [ ] ui-eval-progress.png
- [ ] ui-results-summary.png
- [ ] ui-metric-cards.png
- [ ] ui-per-case-results.png
- [ ] ui-explainability.png
- [ ] ui-retrieval-trace.png
- [ ] ui-export-results.png
- [ ] ui-comparison.png
- [ ] ui-set-baseline.png
- [ ] ui-trends-dashboard.png
- [ ] ui-trend-filters.png

### Low Priority (1 image)

- [ ] security-network-architecture.png

---

**Total: 77 images**

- High Priority: 15 screenshots + 2 branding
- Medium Priority: 61 screenshots/diagrams
- Low Priority: 1 diagram
