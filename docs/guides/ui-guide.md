# Web Platform UI Guide

> **A visual tour of the RAG Evaluator Platform interface.**

This guide walks you through every screen and feature of the web platform, helping you understand the complete workflow from project creation to result analysis.

---

## Table of Contents

- [Dashboard Overview](#dashboard-overview)
- [Project Management](#project-management)
- [Knowledge Base Management](#knowledge-base-management)
- [Test Set Management](#test-set-management)
- [RAG Configuration](#rag-configuration)
- [Running Evaluations](#running-evaluations)
- [Analyzing Results](#analyzing-results)
- [Comparisons & Trends](#comparisons--trends)
- [Keyboard Shortcuts](#keyboard-shortcuts)

---

## Dashboard Overview

The dashboard is your home screen, providing a quick overview of all projects and recent activity.

![Dashboard Overview](../images/ui-dashboard-overview.png)
<!-- PLACEHOLDER: ui-dashboard-overview.png - Full dashboard screenshot showing projects, recent evaluations, quick stats -->

### Dashboard Components

| Component | Description |
|-----------|-------------|
| **Project Cards** | Quick access to all projects with status indicators |
| **Recent Evaluations** | Last 5 evaluations across all projects |
| **Quick Stats** | Total projects, evaluations, and success rate |
| **Quick Actions** | Create project, import, settings |

### Navigation

The sidebar provides access to:

- **Dashboard** - Home overview
- **Projects** - All projects list
- **Settings** - Platform configuration

---

## Project Management

### Creating a Project

![Create Project](../images/ui-create-project.png)
<!-- PLACEHOLDER: ui-create-project.png - Create project dialog -->

1. Click **"+ New Project"** on the dashboard
2. Fill in project details:
   - **Name** (required): A descriptive name
   - **Description** (optional): Project goals or notes
   - **Tags** (optional): For organization (e.g., "production", "experiment")
3. Click **"Create"**

### Project Detail View

![Project Detail](../images/ui-project-detail.png)
<!-- PLACEHOLDER: ui-project-detail.png - Full project detail page showing all sections -->

The project detail page has four main sections:

#### 1. Knowledge Bases

Manage your document collections:

![Knowledge Bases Section](../images/ui-kb-section.png)
<!-- PLACEHOLDER: ui-kb-section.png - Knowledge bases list with cards -->

- **Add Knowledge Base**: Create a new document collection
- **Upload Documents**: Add PDFs, DOCX, TXT files
- **Build Index**: Create searchable indexes for different RAG types

#### 2. Test Sets

Manage evaluation test cases:

![Test Sets Section](../images/ui-testsets-section.png)
<!-- PLACEHOLDER: ui-testsets-section.png - Test sets list -->

- **Create Test Set**: Define question/answer pairs
- **Import/Export**: JSON format support
- **Generate**: AI-powered test generation from KB

#### 3. RAG Configurations

Define RAG settings:

![RAG Configs Section](../images/ui-ragconfigs-section.png)
<!-- PLACEHOLDER: ui-ragconfigs-section.png - RAG config list with type badges -->

- **Create Config**: Choose RAG type and parameters
- **LLM Settings**: Model and provider selection
- **Parameters**: Type-specific configuration

#### 4. Evaluations

View all evaluations:

![Evaluations Section](../images/ui-evaluations-section.png)
<!-- PLACEHOLDER: ui-evaluations-section.png - Evaluations list with status badges -->

- **Status Badges**: Running, Completed, Failed
- **Quick Metrics**: Overview scores
- **Actions**: View, Compare, Retry

### Editing a Project

![Edit Project](../images/ui-edit-project.png)
<!-- PLACEHOLDER: ui-edit-project.png - Edit project dialog -->

1. Click the **Edit** button (pencil icon) in the project header
2. Modify name, description, or tags
3. Click **"Save Changes"**

### Archiving/Deleting

- **Archive**: Hides project from main view (recoverable)
- **Delete**: Permanently removes project and all data

---

## Knowledge Base Management

### Creating a Knowledge Base

![Create KB](../images/ui-create-kb.png)
<!-- PLACEHOLDER: ui-create-kb.png - Create knowledge base dialog -->

1. Click **"+ Add Knowledge Base"**
2. Enter a name and optional description
3. Click **"Create"**

### Uploading Documents

![Upload Documents](../images/ui-upload-docs.png)
<!-- PLACEHOLDER: ui-upload-docs.png - Document upload interface with drag-drop zone -->

Supported formats:
- **PDF** - Portable Document Format
- **DOCX** - Microsoft Word
- **TXT** - Plain text
- **MD** - Markdown

**Upload methods:**
- Drag and drop files onto the upload zone
- Click to browse and select files
- Upload multiple files at once

### Document Management

![Document List](../images/ui-document-list.png)
<!-- PLACEHOLDER: ui-document-list.png - List of documents with metadata -->

For each document, you can:
- View file info (name, size, upload date)
- Preview content
- Remove from knowledge base

### Building Indexes

![Build Index](../images/ui-build-index.png)
<!-- PLACEHOLDER: ui-build-index.png - Index building wizard showing RAG type selection -->

1. Click **"Build Index"** on the knowledge base
2. Select RAG type:
   - **Vector Semantic** - Basic semantic search
   - **Hybrid** - Dense + sparse vectors
   - **Graph RAG** - Knowledge graph
   - **Filesystem RAG** - Agentic navigation
3. Configure parameters (optional)
4. Click **"Start Indexing"**

### Index Progress

![Index Progress](../images/ui-index-progress.png)
<!-- PLACEHOLDER: ui-index-progress.png - Indexing progress bar with status -->

The progress view shows:
- Current phase (loading, chunking, embedding, storing)
- Progress percentage
- Estimated time remaining
- Cancel option

### Index Detail

![Index Detail](../images/ui-index-detail.png)
<!-- PLACEHOLDER: ui-index-detail.png - Completed index with stats -->

After indexing completes:
- **Chunk Count**: Number of indexed chunks
- **Index Size**: Storage used
- **Build Time**: How long indexing took
- **Status**: Ready, Failed, Building

---

## Test Set Management

### Creating a Test Set

![Create Test Set](../images/ui-create-testset.png)
<!-- PLACEHOLDER: ui-create-testset.png - Create test set dialog -->

1. Click **"+ Add Test Set"**
2. Enter name and description
3. Click **"Create"**

### Adding Test Cases

![Add Test Case](../images/ui-add-testcase.png)
<!-- PLACEHOLDER: ui-add-testcase.png - Add test case form -->

For each test case:
- **Question**: The query to evaluate
- **Expected Answer**: The ground truth answer
- **Difficulty** (optional): easy, medium, hard
- **Tags** (optional): For filtering

### Bulk Import

![Import Test Cases](../images/ui-import-testcases.png)
<!-- PLACEHOLDER: ui-import-testcases.png - JSON import dialog -->

Import from JSON format:

```json
[
  {
    "question": "What is RAG?",
    "expected_answer": "RAG stands for Retrieval-Augmented Generation...",
    "difficulty": "easy",
    "tags": ["definition"]
  }
]
```

### AI-Powered Generation

![Generate Test Cases](../images/ui-generate-testcases.png)
<!-- PLACEHOLDER: ui-generate-testcases.png - Test generation wizard -->

1. Click **"Generate from KB"**
2. Select the knowledge base to analyze
3. Choose difficulty distribution
4. Set number of questions to generate
5. Click **"Generate"**

### Reviewing Generated Cases

![Review Generated](../images/ui-review-generated.png)
<!-- PLACEHOLDER: ui-review-generated.png - Generated test cases with approve/reject buttons -->

Generated cases require review:
- **Approve**: Add to test set
- **Edit**: Modify before adding
- **Reject**: Discard

---

## RAG Configuration

### Creating a RAG Config

![Create RAG Config](../images/ui-create-ragconfig.png)
<!-- PLACEHOLDER: ui-create-ragconfig.png - RAG config creation wizard -->

#### Step 1: Basic Info

- **Name**: Descriptive configuration name
- **RAG Type**: Select implementation

#### Step 2: LLM Settings

![LLM Settings](../images/ui-llm-settings.png)
<!-- PLACEHOLDER: ui-llm-settings.png - LLM provider and model selection -->

- **Provider**: OpenAI, Anthropic, Ollama, etc.
- **Model**: Specific model selection
- **Temperature**: Generation randomness (0.0-1.0)

#### Step 3: RAG Parameters

Parameters vary by RAG type:

**Vector Semantic:**
![Vector Params](../images/ui-vector-params.png)
<!-- PLACEHOLDER: ui-vector-params.png - Vector semantic parameters -->

- Collection name
- Persist directory

**Hybrid Search:**
![Hybrid Params](../images/ui-hybrid-params.png)
<!-- PLACEHOLDER: ui-hybrid-params.png - Hybrid search parameters -->

- Qdrant URL
- Collection name
- Fusion alpha

**Graph RAG:**
![Graph Params](../images/ui-graph-params.png)
<!-- PLACEHOLDER: ui-graph-params.png - Graph RAG parameters -->

- Neo4j URI
- Credentials
- Vector index name

**Filesystem RAG:**
![Filesystem Params](../images/ui-filesystem-params.png)
<!-- PLACEHOLDER: ui-filesystem-params.png - Filesystem RAG parameters -->

- Max iterations
- Max tool calls
- Word threshold

---

## Running Evaluations

### Start Evaluation Wizard

![Start Evaluation](../images/ui-start-evaluation.png)
<!-- PLACEHOLDER: ui-start-evaluation.png - Full evaluation wizard -->

#### Step 1: Select Components

- **Knowledge Base**: Choose indexed KB
- **Index**: Select specific index (if multiple)
- **Test Set**: Choose test cases to run
- **RAG Config**: Select configuration

#### Step 2: Choose Metrics

![Choose Metrics](../images/ui-choose-metrics.png)
<!-- PLACEHOLDER: ui-choose-metrics.png - Metric selection checkboxes -->

Select which metrics to evaluate:
- Faithfulness
- Answer Relevancy
- Contextual Precision
- Contextual Recall
- Correctness (G-Eval)

#### Step 3: Review & Run

![Review Run](../images/ui-review-run.png)
<!-- PLACEHOLDER: ui-review-run.png - Review summary before starting -->

- Confirm all selections
- Estimated time and cost
- Click **"Start Evaluation"**

### Evaluation Progress

![Evaluation Progress](../images/ui-eval-progress.png)
<!-- PLACEHOLDER: ui-eval-progress.png - Real-time progress view -->

Real-time progress shows:
- **Current Test Case**: Question being evaluated
- **Progress Bar**: Completed / Total
- **Live Metrics**: Running average scores
- **Token Usage**: API consumption
- **Cancel**: Stop evaluation

### Controlling Evaluations

| Action | Description |
|--------|-------------|
| **Pause** | Suspend and save checkpoint |
| **Resume** | Continue from checkpoint |
| **Cancel** | Stop and discard results |
| **Retry** | Re-run failed evaluation |

---

## Analyzing Results

### Results Summary

![Results Summary](../images/ui-results-summary.png)
<!-- PLACEHOLDER: ui-results-summary.png - Complete results summary with all metric cards -->

The summary view shows:
- **Metric Cards**: Score for each metric with pass/fail indicator
- **Distribution Chart**: Score distribution across test cases
- **Quick Stats**: Pass rate, average scores, token usage

### Metric Cards

![Metric Cards](../images/ui-metric-cards.png)
<!-- PLACEHOLDER: ui-metric-cards.png - Close-up of metric cards with scores -->

Each card displays:
- **Score**: 0.0 - 1.0 value
- **Status**: Pass/Fail based on threshold
- **Trend**: Comparison to baseline (if set)

### Per-Case Results

![Per-Case Results](../images/ui-per-case-results.png)
<!-- PLACEHOLDER: ui-per-case-results.png - Expandable results table -->

Click any row to expand:
- Full question and answer
- Retrieved context chunks
- Individual metric scores
- Explainability button

### Metric Explainability

![Explainability Panel](../images/ui-explainability.png)
<!-- PLACEHOLDER: ui-explainability.png - Metric explanation panel -->

Click **"Why this score?"** to see:
- LLM judge reasoning
- Claims analysis (for Faithfulness)
- Supporting evidence
- Improvement suggestions

### Retrieval Trace

![Retrieval Trace](../images/ui-retrieval-trace.png)
<!-- PLACEHOLDER: ui-retrieval-trace.png - Detailed retrieval trace view -->

The trace viewer shows:
- **Query**: Original question
- **Strategy**: Retrieval method used
- **Chunks**: Retrieved chunks with scores
- **Timeline**: Processing steps with durations

### Exporting Results

![Export Options](../images/ui-export-results.png)
<!-- PLACEHOLDER: ui-export-results.png - Export dropdown menu -->

Export formats:
- **JSON**: Complete structured data
- **Markdown**: Human-readable report
- **CSV**: Per-case results for spreadsheets

---

## Comparisons & Trends

### Comparing Evaluations

![Comparison View](../images/ui-comparison.png)
<!-- PLACEHOLDER: ui-comparison.png - Side-by-side evaluation comparison -->

Compare multiple evaluations:
1. Select evaluations to compare
2. View side-by-side metrics
3. Identify winning configuration

### Setting a Baseline

![Set Baseline](../images/ui-set-baseline.png)
<!-- PLACEHOLDER: ui-set-baseline.png - Baseline selection interface -->

1. Click **"Set as Baseline"** on an evaluation
2. Future evaluations show delta vs. baseline
3. Track improvements over time

### Trends Dashboard

![Trends Dashboard](../images/ui-trends-dashboard.png)
<!-- PLACEHOLDER: ui-trends-dashboard.png - Full trends view with charts -->

The trends view shows:
- **Line Charts**: Metrics over time
- **Grouped View**: By RAG type or configuration
- **Cost Tracking**: Token usage trends
- **Efficiency Map**: Score vs. cost visualization

### Trend Filters

![Trend Filters](../images/ui-trend-filters.png)
<!-- PLACEHOLDER: ui-trend-filters.png - Filter options for trends -->

Filter by:
- Date range
- RAG type
- Test set
- Metric

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + N` | New project |
| `Ctrl/Cmd + E` | Start evaluation |
| `Ctrl/Cmd + S` | Save current form |
| `Esc` | Close dialog |
| `?` | Show shortcuts help |

---

## Tips & Best Practices

### Organizing Projects

- Use tags to categorize (e.g., "production", "experiment", "v2")
- Keep related KBs and test sets in the same project
- Archive completed experiments

### Efficient Evaluations

- Start with fewer metrics during development
- Use async mode for faster results
- Set appropriate thresholds

### Result Analysis

- Always check low-scoring cases
- Use explainability for debugging
- Compare against baseline regularly

---

## Related Documentation

- [Getting Started Guide](getting-started.md) - First steps
- [Evaluation Guide](evaluation-guide.md) - Deep dive into evaluation
- [API Reference](../api.md) - Programmatic access
- [Troubleshooting](troubleshooting.md) - Common issues
