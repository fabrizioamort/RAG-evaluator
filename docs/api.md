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
