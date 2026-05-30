# API Reference

The platform backend is a FastAPI application. The OpenAPI schema is available at
`http://localhost:8000/api/v1/docs` when the backend is running.

Base path:

```text
/api/v1
```

The open source edition does not include built-in authentication. Protect the API with
a reverse proxy, private network, or API gateway before exposing it beyond a local
development environment.

## Response Conventions

List endpoints return paginated objects:

```json
{
  "items": [],
  "total": 0,
  "offset": 0,
  "limit": 50
}
```

Errors use a structured response with `detail`, `request_id`, and optional validation
details. Every request receives an `X-Request-ID` response header.

## Health And Dashboard

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Basic health check |
| GET | `/health/detail` | Detailed service health |
| GET | `/stats` | Dashboard counters |
| GET | `/recent-activity` | Recent platform activity |

## Projects

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects` | List projects. Supports `limit`, `offset`, and `status`. |
| POST | `/projects` | Create a project. |
| GET | `/projects/{project_id}` | Get project details. |
| PUT | `/projects/{project_id}` | Update project metadata. |
| DELETE | `/projects/{project_id}` | Delete a project and related data. |
| POST | `/projects/{project_id}/archive` | Archive a project. |
| GET | `/projects/{project_id}/baseline` | Get the current baseline evaluation. |

Create request:

```json
{
  "name": "Internal docs RAG",
  "description": "Evaluate support documentation retrieval",
  "tags": ["support", "baseline"]
}
```

## Knowledge Bases And Documents

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects/{project_id}/knowledge-bases` | List knowledge bases in a project. |
| POST | `/projects/{project_id}/knowledge-bases` | Create a knowledge base. |
| GET | `/knowledge-bases/{kb_id}` | Get a knowledge base with documents. |
| PUT | `/knowledge-bases/{kb_id}` | Update knowledge base metadata. |
| DELETE | `/knowledge-bases/{kb_id}` | Delete a knowledge base. |
| POST | `/knowledge-bases/{kb_id}/archive` | Archive a knowledge base. |
| POST | `/knowledge-bases/{kb_id}/restore` | Restore an archived knowledge base. |
| POST | `/knowledge-bases/{kb_id}/documents` | Upload documents using `multipart/form-data` field `files`. |
| DELETE | `/knowledge-bases/{kb_id}/documents/{doc_id}` | Delete a document. |
| GET | `/knowledge-bases/{kb_id}/versions` | List knowledge base versions. |
| GET | `/knowledge-bases/{kb_id}/status` | Get processing/indexing status. |

Supported upload formats include PDF, DOCX, TXT, and Markdown where the active loader
supports them.

## Indexes

Indexes are isolated physical builds of a knowledge base using a specific RAG
configuration. Evaluations run against indexes, not directly against mutable knowledge
bases.

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/indexes` | List indexes. Supports `kb_id`, `project_id`, `status`, `limit`, and `offset`. |
| POST | `/knowledge-bases/{kb_id}/indexes` | Create and start an index build. |
| GET | `/indexes/{index_id}` | Get index details. |
| DELETE | `/indexes/{index_id}` | Delete an index and clean up storage when possible. |
| GET | `/indexes/{index_id}/stream` | SSE stream for build progress. |
| POST | `/indexes/{index_id}/retry` | Retry a failed index build. |
| POST | `/indexes/{index_id}/archive` | Archive an index while preserving evaluation references. |

Create request:

```json
{
  "rag_config_id": "00000000-0000-0000-0000-000000000000",
  "name": "Hybrid support docs index",
  "description": "Qdrant hybrid index for support documentation"
}
```

## RAG Configurations

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/rag-types` | List supported RAG types and parameter metadata. |
| GET | `/rag-types/{rag_type}/parameters` | Get parameter schema for one RAG type. |
| GET | `/llm-providers` | List provider/model options exposed by the platform. |
| GET | `/projects/{project_id}/rag-configs` | List RAG configurations. |
| POST | `/projects/{project_id}/rag-configs` | Create a RAG configuration. |
| GET | `/rag-configs/{config_id}` | Get a RAG configuration. |
| PUT | `/rag-configs/{config_id}` | Update a RAG configuration. |
| DELETE | `/rag-configs/{config_id}` | Delete a RAG configuration. |

RAG type parameter metadata includes `phase` (`build` or `query`) and
`platform_managed`. Build-phase parameters are captured when an index is built and
cannot be overridden while querying a ready index. Query-phase parameters can be
overridden on evaluation and playground requests.

Create request:

```json
{
  "name": "Hybrid gpt-5-mini",
  "rag_type": "vector_hybrid",
  "llm_provider": "openai",
  "llm_model": "gpt-5-mini",
  "embedding_model": "text-embedding-3-small",
  "llm_base_url": null,
  "parameters": {
    "qdrant_url": "http://localhost:6333",
    "sparse_model_name": "prithvida/Splade_pp_en_v1"
  }
}
```

Supported RAG types:

| Type | Build-time parameters | Query-time parameters |
| --- | --- | --- |
| `vector_semantic` | `embedding_model`, `collection_name`, `persist_directory`, chunking controls | `llm_model`, `top_k` |
| `vector_hybrid` | `embedding_model`, `sparse_model_name`, `collection_name`, `qdrant_url`, chunking controls | `llm_model`, `top_k` |
| `graph_rag` | `embedding_model`, `extraction_model`, `neo4j_uri`, `neo4j_username`, `neo4j_password`, `vector_index_name` | `llm_model`, `top_k` |
| `filesystem_rag` | `prepared_path`, `word_threshold` | `llm_model`, `max_iterations`, `max_tool_calls`, `max_file_reads` |
| `rlm_rag` | `chunk_size`, `chunk_overlap`, `use_llm_summaries`, `use_llm_topics`, `max_topics_per_doc`, `worker_model`, prepared path | `llm_model`, `orchestrator_model`, `security_mode`, `max_repl_steps`, `repl_timeout`, `max_file_reads`, `max_read_bytes`, `max_read_lines`, `max_sub_calls`, `max_recursion_depth`, `small_corpus_threshold`, `top_k` |

Supported provider metadata currently includes OpenAI, OpenRouter, Anthropic, and
Ollama. Provider support depends on configured API keys and local services.

## Test Sets And Test Generation

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects/{project_id}/test-sets` | List test sets. |
| POST | `/projects/{project_id}/test-sets` | Create an empty test set. |
| GET | `/test-sets/{test_set_id}` | Get a test set with cases. |
| PUT | `/test-sets/{test_set_id}` | Update test set metadata. |
| DELETE | `/test-sets/{test_set_id}` | Delete a test set. |
| POST | `/test-sets/{test_set_id}/cases` | Add one test case. |
| POST | `/test-sets/{test_set_id}/cases/bulk` | Add multiple test cases. |
| PUT | `/test-sets/{test_set_id}/cases/{case_id}` | Update a test case. |
| DELETE | `/test-sets/{test_set_id}/cases/{case_id}` | Delete a test case. |
| POST | `/test-sets/{test_set_id}/cases/bulk-review` | Approve or reject generated cases. |
| POST | `/projects/{project_id}/test-sets/import` | Import a test set from JSON. |
| GET | `/test-sets/{test_set_id}/export` | Export a test set to JSON. |
| POST | `/test-sets/{test_set_id}/generate` | Start AI-assisted test generation. |
| GET | `/test-sets/{test_set_id}/generation-status` | Get latest generation job status. |
| DELETE | `/test-sets/{test_set_id}/generation` | Cancel a running generation job. |
| GET | `/test-sets/{test_set_id}/generation-jobs` | List generation jobs for the test set. |

Import request:

```json
{
  "name": "Support smoke tests",
  "description": "A small hand-reviewed test set",
  "tags": ["smoke"],
  "test_cases": [
    {
      "question": "How do users reset a password?",
      "expected_answer": "Users reset a password from the account settings page.",
      "ground_truth_context": ["Password reset instructions are in account settings."],
      "difficulty": "easy",
      "category": "support",
      "question_type": "factual"
    }
  ]
}
```

Generation request:

```json
{
  "knowledge_base_id": "00000000-0000-0000-0000-000000000000",
  "target_count": 20,
  "questions_per_chunk": 1,
  "difficulty_distribution": {
    "easy": 0.3,
    "medium": 0.5,
    "hard": 0.2
  },
  "skip_semantic_check": false
}
```

## Test Templates

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/test-templates` | List built-in and custom templates. |
| POST | `/test-templates` | Create a template. |
| GET | `/test-templates/{template_id}` | Get a template. |
| PUT | `/test-templates/{template_id}` | Update a template. |
| DELETE | `/test-templates/{template_id}` | Delete a template. |

## Evaluations

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/evaluations` | Start an evaluation against a ready index. |
| GET | `/evaluations/{evaluation_id}` | Get evaluation summary. |
| PATCH | `/evaluations/{evaluation_id}` | Update evaluation name, notes, or tags. |
| GET | `/evaluations/{evaluation_id}/results` | List per-question results. |
| GET | `/evaluations/{evaluation_id}/trace/{result_id}` | Get retrieval trace artifact for one result. |
| GET | `/evaluations/{evaluation_id}/manifest` | Get reproducibility snapshot. |
| GET | `/evaluations/{evaluation_id}/stream` | SSE stream for progress. |
| POST | `/evaluations/{evaluation_id}/cancel` | Cancel a running evaluation. |
| POST | `/evaluations/{evaluation_id}/pause` | Pause a running evaluation. |
| POST | `/evaluations/{evaluation_id}/resume` | Resume a paused evaluation. |
| POST | `/evaluations/{evaluation_id}/retry` | Retry a failed or cancelled evaluation. |
| POST | `/evaluations/{evaluation_id}/set-baseline` | Mark a completed evaluation as project baseline. |
| GET | `/projects/{project_id}/evaluations` | List evaluations in a project. Supports `status`. |

Create request:

```json
{
  "name": "Hybrid baseline run",
  "knowledge_base_index_id": "00000000-0000-0000-0000-000000000000",
  "test_set_id": "11111111-1111-1111-1111-111111111111",
  "metric_names": ["faithfulness", "relevancy", "precision", "recall", "g_eval"],
  "query_overrides": {
    "llm_model": "gpt-5-mini",
    "top_k": 8,
    "parameters": {}
  },
  "eval_judge_model": "gpt-5-mini",
  "include_reason": true,
  "notes": "Initial full metric run",
  "tags": ["baseline"]
}
```

`query_overrides` may include `llm_model`, `top_k`, and query-phase entries in
`parameters`. Build-phase overrides such as `embedding_model`, chunking, storage paths,
sparse model, graph extraction model, or preparation controls are rejected for ready
indexes. If `eval_judge_model` is omitted, the backend defaults the judge to the
effective RAG generation model.

Run manifests include the legacy `rag_config_snapshot` plus:

- `build_config_snapshot`: immutable settings used to build the selected index.
- `query_overrides`: per-run query overrides requested by the user.
- `effective_config_snapshot`: build snapshot plus query overrides resolved for the run.
- `generation_model` and `eval_judge_model`.

Evaluation statuses are `pending`, `running`, `completed`, `failed`, `cancelled`, and
`paused`.

## Comparisons

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/comparisons` | Create a comparison from one baseline evaluation and one or more compared evaluations. |
| GET | `/comparisons/{comparison_id}` | Get aggregate and per-question comparison details. |
| DELETE | `/comparisons/{comparison_id}` | Delete a comparison. |
| GET | `/projects/{project_id}/comparisons` | List comparisons in a project. |
| GET | `/evaluations/{evaluation_id}/comparisons` | List comparisons involving one evaluation. |

Create request:

```json
{
  "name": "Semantic vs hybrid",
  "description": "Compare the two completed baseline candidates",
  "baseline_evaluation_id": "00000000-0000-0000-0000-000000000000",
  "compared_evaluation_ids": [
    "11111111-1111-1111-1111-111111111111"
  ]
}
```

## Trends

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects/{project_id}/trends` | Metric history grouped by RAG config. |
| GET | `/rag-configs/{rag_config_id}/trends` | Metric history for one RAG config. |

## Playground

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/playground/indexes` | List ready indexes available for ad hoc querying. |
| POST | `/playground/query` | Query one to four indexes and save the result. |
| GET | `/playground/history` | List saved playground queries. |
| GET | `/playground/history/{query_id}` | Get one saved query with all results. |
| DELETE | `/playground/history/{query_id}` | Delete a saved query. |

Query request:

```json
{
  "question": "What are the main support escalation steps?",
  "index_ids": [
    "00000000-0000-0000-0000-000000000000"
  ],
  "top_k": 5,
  "query_overrides": {
    "llm_model": "gpt-5-mini",
    "top_k": 5,
    "parameters": {}
  }
}
```

The top-level `top_k` is retained for compatibility. When `query_overrides.top_k` is
provided, it is used as the effective query execution value and is recorded with the
saved playground query.

## Webhooks

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects/{project_id}/webhooks` | List project webhooks. |
| POST | `/projects/{project_id}/webhooks` | Create a webhook. |
| GET | `/webhooks/{webhook_id}` | Get webhook details. |
| PATCH | `/webhooks/{webhook_id}` | Update webhook configuration. |
| DELETE | `/webhooks/{webhook_id}` | Delete a webhook. |
| POST | `/webhooks/{webhook_id}/test` | Send a test payload. |

The default maximum is three webhooks per project.

## Example Requests

```powershell
# Create a project
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/v1/projects `
  -ContentType "application/json" `
  -Body '{"name":"Demo","description":"RAG evaluation demo"}'

# Check health
Invoke-RestMethod http://localhost:8000/api/v1/health
```
