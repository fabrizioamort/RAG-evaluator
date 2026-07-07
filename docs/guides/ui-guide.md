# Web UI Guide

The web platform is the primary workflow for managing RAG experiments. It runs as a
React application on `http://localhost:3000` and calls the FastAPI backend on
`http://localhost:8000/api/v1`.

## Main Navigation

| Route | Purpose |
| --- | --- |
| `/` | Dashboard with project and activity overview. |
| `/projects` | Project list. |
| `/projects/{id}` | Project workspace with knowledge bases, test sets, RAG configs, evaluations, comparisons, and trends. |
| `/knowledge-bases/{id}` | Knowledge base detail and document management. |
| `/indexes` | Cross-project index list. |
| `/indexes/{id}` | Index detail and build status. |
| `/playground` | Ad hoc querying against ready indexes. |

## Project Workflow

Every evaluation belongs to a project. A project groups the documents, test sets,
RAG configurations, indexes, evaluations, comparisons, and trends for one use case.

Recommended workflow:

1. Create a project.
2. Create a knowledge base and upload documents.
3. Create one or more RAG configurations.
4. Build indexes from the knowledge base and RAG configurations.
5. Create, import, or generate a test set.
6. Run evaluations against ready indexes.
7. Compare completed evaluations and mark a baseline.
8. Use trends to track changes over time.

## Knowledge Bases

Knowledge bases hold source documents. From the Knowledge Bases tab you can:

- Create a knowledge base.
- Upload documents.
- Open a knowledge base detail page.
- Build indexes for different RAG configurations.

Document uploads use the backend storage service. Uploaded files are copied into
`STORAGE_PATH/documents` and associated with the knowledge base. The knowledge base
detail page lists documents with pagination and filename search, so large uploads can
be inspected without loading the full document set into one view.

## Indexes

An index is a physical build of a knowledge base for one RAG configuration. This
separation is important: evaluations run against immutable-ish index snapshots instead
of the current mutable knowledge base.

Index states:

| State | Meaning |
| --- | --- |
| `pending` | Created and waiting for the build task. |
| `building` | Build is in progress. |
| `ready` | Ready for playground queries and evaluations. |
| `failed` | Build failed and can be retried. |
| `archived` | Hidden from normal active workflows but retained for history. |

Build progress is streamed from the backend with Server-Sent Events. Failed indexes can
be retried. The global Indexes page supports cross-project status filtering, project
filtering, URL-backed search, pagination, and launching an evaluation directly from a
ready index. Indexes referenced by evaluations should usually be archived instead of
deleted.

## RAG Configurations

RAG configurations define retrieval strategy, provider/model settings, and strategy
parameters.

Available strategies:

- Vector Semantic Search (`vector_semantic`)
- Hybrid Search (`vector_hybrid`)
- Graph RAG (`graph_rag`)
- Filesystem RAG (`filesystem_rag`)
- RLM-RAG (`rlm_rag`)
- Google Vertex AI Search (`google_vertex_search`)

The form loads parameter metadata from the backend. For platform-managed indexes, leave
storage fields blank unless you are intentionally reusing external storage.

## Test Sets

Test sets contain the questions and expected answers used by evaluations.

You can:

- Create a blank test set.
- Add or edit test cases manually.
- Import a JSON test set.
- Export a test set.
- Generate test cases from a knowledge base.
- Review, approve, or reject generated cases in bulk.

Import shape:

```json
{
  "name": "Smoke tests",
  "description": "Initial validation cases",
  "tags": ["smoke"],
  "test_cases": [
    {
      "question": "What does the refund policy allow?",
      "expected_answer": "Customers can request a refund within the stated policy window.",
      "ground_truth_context": ["Refund policy source text."],
      "difficulty": "medium",
      "category": "policy",
      "question_type": "factual"
    }
  ]
}
```

Generated test cases are not automatically trusted. Review them before using them for
baseline or release decisions.

## Evaluations

An evaluation runs selected metrics for every test case in a test set against one ready
index.

The start wizard asks for:

- Knowledge base index.
- Test set.
- Metric list.
- Optional name, notes, and tags.

Available metric names:

- `faithfulness`
- `relevancy`
- `precision`
- `recall`
- `g_eval`

The evaluation progress view streams updates while the backend runner executes. You can
cancel, pause, resume, or retry evaluations depending on the current state.

The project Evaluations tab supports URL-backed filtering by test set, RAG
configuration, index, and status. Evaluation lists are paginated so large projects do
not need to load every run at once.

## Result Analysis

Completed evaluations include:

- Average metric scores.
- Overall pass rate.
- Per-question generated answers.
- Expected answers.
- LLM judge reasons when enabled.
- Retrieval traces.
- Cost and token usage.
- Latency metrics.
- Run manifest snapshots.

Use low-scoring cases to diagnose whether the failure came from retrieval, generation,
or test-set quality. Use retrieval traces to inspect which chunks were retrieved and in
which order.

Evaluation result lists are loaded page by page and can be searched by question,
expected answer, or generated answer. The run manifest view separates the immutable
build snapshot, query overrides, and effective configuration used for the run.

## Comparisons

The Comparisons tab compares one baseline evaluation against one or more completed
evaluations from the same project.

Comparison output includes:

- Metric deltas.
- Cost and latency differences.
- Configuration differences.
- Per-question result deltas.

Use comparisons when you change only one major variable at a time, such as RAG type,
chunk size, model, or provider. This makes the result easier to interpret.

## Trends

The Trends tab visualizes evaluation history for a project. Use it to track regressions,
quality improvements, and efficiency changes across RAG configurations.

Trend views include metric history and efficiency-oriented cost/latency views where
data is available.

## Playground

The playground lets you query up to four ready indexes at once without creating a test
set or evaluation. It is useful for quick qualitative checks before spending time and
tokens on a full run.

Playground results include:

- Answer per index.
- Retrieved context.
- Retrieval trace details.
- Latency and token metrics.
- Saved query history.

## Practical Tips

- Start with a small document set and a small test set.
- Build `vector_semantic` first as a baseline.
- Use the playground before a full evaluation when testing a new RAG configuration.
- Keep baseline evaluations stable and compare against them.
- Prefer archiving over deleting when results need to remain reproducible.
- Review generated test cases before they influence quality decisions.
