# API Reference - RAG Evaluation Platform

The RAG Evaluation Platform provides a RESTful API for managing projects, knowledge bases, test sets, RAG configurations, and running evaluations.

## Base URL

All API requests are prefixed with:
`/api/v1`

## API Endpoints

### Projects

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects` | List all projects (supports `?status=`, `?tags=`) |
| POST | `/projects` | Create a new project |
| GET | `/projects/{id}` | Get project details |
| PUT | `/projects/{id}` | Update project information |
| DELETE | `/projects/{id}` | Delete a project (cascades to all related data) |
| POST | `/projects/{id}/archive` | Archive a project |

### Knowledge Bases

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects/{pid}/knowledge-bases` | List knowledge bases in a project |
| POST | `/projects/{pid}/knowledge-bases` | Create a new knowledge base |
| GET | `/knowledge-bases/{id}` | Get knowledge base details |
| DELETE | `/knowledge-bases/{id}` | Delete a knowledge base |
| POST | `/knowledge-bases/{id}/documents` | Upload documents (multipart/form-data) |
| DELETE | `/knowledge-bases/{id}/documents/{docId}` | Remove a document |
| POST | `/knowledge-bases/{id}/index` | Trigger indexing of the knowledge base |
| GET | `/knowledge-bases/{id}/status` | Get indexing status |
| GET | `/knowledge-bases/{id}/versions` | List knowledge base versions |

### Test Sets

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects/{pid}/test-sets` | List test sets in a project |
| POST | `/projects/{pid}/test-sets` | Create a new test set |
| GET | `/test-sets/{id}` | Get test set details and its test cases |
| PUT | `/test-sets/{id}` | Update test set information |
| DELETE | `/test-sets/{id}` | Delete a test set |
| POST | `/test-sets/{id}/cases` | Add a test case to the set |
| PUT | `/test-sets/{id}/cases/{caseId}` | Update a test case |
| DELETE | `/test-sets/{id}/cases/{caseId}` | Delete a test case |
| POST | `/test-sets/{id}/import` | Import test cases from a JSON file |
| GET | `/test-sets/{id}/export` | Export test cases to JSON |
| POST | `/test-sets/{id}/generate` | Generate test cases from a knowledge base |
| GET | `/test-sets/{id}/generation-status` | Get the status of a generation job |
| POST | `/test-sets/{id}/cases/bulk-review` | Bulk approve or reject generated cases |

### RAG Configurations

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects/{pid}/rag-configs` | List RAG configurations in a project |
| POST | `/projects/{pid}/rag-configs` | Create a new RAG configuration |
| GET | `/rag-configs/{id}` | Get configuration details |
| PUT | `/rag-configs/{id}` | Update a RAG configuration |
| DELETE | `/rag-configs/{id}` | Delete a configuration |
| GET | `/rag-types` | List available RAG types (vector_semantic, hybrid, etc.) |
| GET | `/rag-types/{type}/parameters` | Get the configuration schema for a specific type |
| GET | `/llm-providers` | List available LLM providers (OpenAI, Ollama, etc.) |

#### RAG Parameters

When creating or updating a RAG config, send a `parameters` object with the fields for the selected type. Use `GET /rag-types` and `GET /rag-types/{type}/parameters` to discover the parameter names, types, defaults, and descriptions. If a parameter is omitted, the backend falls back to environment settings (from `.env`) or internal defaults. When you build an index through the platform, collection and storage paths are auto-managed for isolation, so storage-related parameters are usually optional.

#### vector_semantic

- `collection_name` (default: `rag_documents`): ChromaDB collection name. In the platform, indexes use a generated collection name, so keep the default unless you need to reuse an existing collection.
- `persist_directory` (default: empty): Filesystem path for Chroma persistence. Leave blank to use platform-managed storage or the core default.

#### vector_hybrid

- `collection_name` (default: empty): Qdrant collection name. In the platform, indexes use a generated collection name, so leave blank unless you need to reuse an existing collection.
- `qdrant_url` (default: empty): Qdrant server URL, for example `http://localhost:6333`. If blank, the backend uses `QDRANT_URL` from `.env` or the core default.

#### graph_rag

- `neo4j_uri` (default: empty): Neo4j connection URI, for example `bolt://localhost:7687`. If blank, the backend uses `NEO4J_URI` from `.env` or the core default.
- `neo4j_username` (default: empty): Neo4j username. If blank, the backend uses `NEO4J_USERNAME` from `.env` or the core default.
- `neo4j_password` (default: empty): Neo4j password. If blank, the backend uses `NEO4J_PASSWORD` from `.env` or the core default.
- `vector_index_name` (default: `chunk_embeddings`): Name of the Neo4j vector index. Keep the default unless you created a custom index.

#### filesystem_rag

- `llm_model` (default: `gpt-4o-mini`): Model used by the agent for navigation. In the platform, this is driven by the LLM Settings section, so the default is usually fine.
- `prepared_path` (default: `data/prepared/filesystem_rag`): Path to prepared filesystem output. The platform stores this under `storage/indexes/<index_id>/filesystem_rag`, so leave blank unless you need a custom path.
- `word_threshold` (default: `1000`): Word count threshold for LLM analysis vs heuristic analysis. Lower values use the LLM more (higher cost).
- `max_iterations` (default: `10`): Max ReAct loop iterations per query.
- `max_tool_calls` (default: `20`): Max tool calls per query.
- `max_file_reads` (default: `10`): Max file reads per query.

### Evaluations

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects/{pid}/evaluations` | List evaluations in a project |
| POST | `/evaluations` | Start a new evaluation |
| GET | `/evaluations/{id}` | Get evaluation details and summary metrics |
| GET | `/evaluations/{id}/results` | Get detailed per-question results (paginated) |
| GET | `/evaluations/{id}/stream` | SSE stream for real-time progress updates |
| POST | `/evaluations/{id}/cancel` | Cancel a running evaluation |
| POST | `/evaluations/{id}/pause` | Pause an evaluation (save checkpoint) |
| POST | `/evaluations/{id}/resume` | Resume from a checkpoint |
| POST | `/evaluations/{id}/retry` | Retry a failed evaluation |
| GET | `/evaluations/{id}/report` | Download the evaluation report (`?format=json\|markdown`) |
| GET | `/evaluations/{id}/manifest` | Get the reproducibility run manifest |
| GET | `/evaluations/{id}/trace/{resultId}` | Get the retrieval trace for a specific result |
| POST | `/evaluations/{id}/set-baseline` | Mark this evaluation as the project baseline |

### Webhooks

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/projects/{pid}/webhooks` | List webhooks for a project |
| POST | `/projects/{pid}/webhooks` | Create a new webhook (max 3 per project) |
| GET | `/webhooks/{id}` | Get webhook details |
| PUT | `/webhooks/{id}` | Update webhook configuration |
| DELETE | `/webhooks/{id}` | Delete a webhook |
| POST | `/webhooks/{id}/test` | Send a test payload to the webhook URL |

## Authentication

In the Open Source Edition, the API is accessible without authentication for local use. For production environments, it is recommended to run the platform behind a reverse proxy with authentication.

## Example Usage

### Creating a Project

```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "My RAG Project", "description": "Evaluating internal documentation"}'
```

### Starting an Evaluation

```bash
curl -X POST http://localhost:8000/api/v1/evaluations \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "...",
    "knowledge_base_id": "...",
    "test_set_id": "...",
    "rag_config_id": "..."
  }'
```
