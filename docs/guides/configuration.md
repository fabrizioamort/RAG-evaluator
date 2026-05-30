# Configuration

RAG Evaluator is configured with environment variables. For most local development,
copy the root example file and edit it:

```powershell
Copy-Item .env.example .env
```

The core CLI reads the root `.env`. The backend reads `platform/backend/.env` when it
exists and also falls back to the root `.env`.

## Minimum Configuration

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini
EMBEDDING_MODEL=text-embedding-3-small
```

For an OpenAI-compatible provider:

```env
OPENAI_API_KEY=your_provider_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## Core And CLI Settings

These variables are used by `uv run rag-eval ...` and by the shared RAG
implementations.

### LLM And Embeddings

| Variable | Default in code/example | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | API key for OpenAI or an OpenAI-compatible provider. |
| `OPENAI_BASE_URL` | unset | Optional custom base URL for OpenAI-compatible APIs. |
| `OPENAI_MODEL` | example: `gpt-5-mini` | Generation and judge model for CLI runs. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for vector stores. |
| `OPENAI_TIMEOUT` | `600` | OpenAI client timeout in seconds. |

### ChromaDB

| Variable | Default | Description |
| --- | --- | --- |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma_db` | Local ChromaDB persistence directory. |

### Qdrant And Hybrid Search

| Variable | Default | Description |
| --- | --- | --- |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint. |
| `QDRANT_COLLECTION_NAME` | `hybrid_rag` | Default CLI collection name. |
| `HYBRID_CHUNK_SIZE` | `700` | Chunk size for hybrid indexing. |
| `HYBRID_CHUNK_OVERLAP` | `100` | Chunk overlap for hybrid indexing. |
| `HYBRID_FUSION_ALPHA` | `0.5` | Dense/sparse weighting where supported by the implementation. |
| `HYBRID_INDEXING_BATCH_SIZE` | `16` | Batch size during hybrid indexing. |
| `SPARSE_MODEL_NAME` | `prithvida/Splade_pp_en_v1` | Sparse embedding model used by FastEmbed. |

### Neo4j And Graph RAG

| Variable | Default | Description |
| --- | --- | --- |
| `NEO4J_AUTH` | `neo4j/password` in the example file | Docker Compose credential pair for the local Neo4j container. |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI. |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username. |
| `NEO4J_PASSWORD` | empty/example value | Neo4j password. |

### Evaluation

| Variable | Default | Description |
| --- | --- | --- |
| `EVAL_TEST_SET_PATH` | `data/test_set.json` | Default CLI test set path. |
| `EVAL_REPORTS_DIR` | `reports` | CLI report output directory. |
| `EVAL_FAITHFULNESS_THRESHOLD` | `0.7` | Pass threshold for faithfulness. |
| `EVAL_ANSWER_RELEVANCY_THRESHOLD` | `0.7` | Pass threshold for answer relevancy. |
| `EVAL_CONTEXTUAL_PRECISION_THRESHOLD` | `0.7` | Pass threshold for contextual precision. |
| `EVAL_CONTEXTUAL_RECALL_THRESHOLD` | `0.7` | Pass threshold for contextual recall. |
| `DEEPEVAL_ASYNC_MODE` | `False` | Enables asynchronous DeepEval execution. |
| `DEEPEVAL_PER_TASK_TIMEOUT` | `900` | Total timeout per DeepEval task in seconds. |
| `DEEPEVAL_PER_ATTEMPT_TIMEOUT` | `300` | Timeout per judge API call attempt. |
| `DEEPEVAL_MAX_RETRIES` | `3` | Retry attempts for judge calls. |
| `DEEPEVAL_MAX_CONCURRENT` | `10` | Maximum concurrent DeepEval tasks. |
| `DEEPEVAL_THROTTLE_VALUE` | `0.0` | Delay between DeepEval calls. |

### Data Directories

| Variable | Default | Description |
| --- | --- | --- |
| `RAW_DATA_DIR` | `data/raw` | Default CLI source document directory. |
| `PROCESSED_DATA_DIR` | `data/processed` | Processed document output directory. |

## Backend Settings

These variables are used by `platform/backend/app/config.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite+aiosqlite:///./storage/dev.db` | Backend database URL. Use PostgreSQL for shared deployments. |
| `STORAGE_PATH` | `./storage` | Base directory for documents, indexes, artifacts, reports, and job logs. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |
| `LOG_FORMAT` | `json` | `json` or `console`. |
| `API_V1_PREFIX` | `/api/v1` | API route prefix. |
| `DEBUG` | `False` | Enables debug mode. |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | JSON array of allowed origins. |
| `OPENAI_API_KEY` | unset | OpenAI key for platform runs. |
| `OPENROUTER_API_KEY` | unset | OpenRouter key. |
| `ANTHROPIC_API_KEY` | unset | Anthropic key. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `DEFAULT_LLM_PROVIDER` | `openai` | Default provider for generated configs and quality checks. |
| `DEFAULT_LLM_MODEL` | `gpt-4o-mini` | Backend default model if not provided by a RAG config. |
| `EVAL_CHECKPOINT_INTERVAL` | `5` | Save progress every N test cases. |
| `EVAL_MAX_CONCURRENT` | `1` | Maximum concurrent evaluations in OSS mode. |
| `EVAL_INCLUDE_REASON` | `True` | Include judge reasoning by default. |
| `EVAL_G_EVAL_THRESHOLD` | `0.7` | Pass threshold for G-Eval correctness. |
| `DEEPEVAL_ASYNC_MODE` | `False` | Enables async metric execution in the backend runner. |
| `DEEPEVAL_MAX_CONCURRENCY` | `5` | Async metric concurrency limit in the backend. |
| `WEBHOOK_MAX_PER_PROJECT` | `3` | Maximum webhooks per project. |
| `WEBHOOK_TIMEOUT_SECONDS` | `30` | Webhook request timeout. |
| `WEBHOOK_MAX_RETRIES` | `3` | Webhook retry count. |

## Database URLs

SQLite for local backend development:

```env
DATABASE_URL=sqlite+aiosqlite:///./storage/dev.db
```

PostgreSQL for shared or production use:

```env
DATABASE_URL=postgresql+asyncpg://rageval:password@localhost:5432/rageval
```

When using the root `docker-compose.yml`, PostgreSQL defaults are:

```env
POSTGRES_USER=rageval
POSTGRES_PASSWORD=rageval
POSTGRES_DB=rageval
```

## Storage Layout

The backend creates these directories under `STORAGE_PATH`:

```text
storage/
  documents/
  indexes/
  artifacts/
  reports/
  logs/
```

Uploaded documents, physical indexes, retrieval traces, generated test-case provenance,
and raw metric artifacts are stored here. Back up this directory together with the
database if you need to preserve evaluation history.

## Provider Selection In The Platform

RAG configurations store provider, default generation model, embedding model, optional
`llm_base_url`, and RAG-specific parameters. The platform currently exposes provider
metadata for:

- OpenAI
- OpenRouter
- Anthropic
- Ollama

Actual availability depends on the matching environment variables and local services.

For web-platform runs, build-time settings are copied into the index snapshot when an
index is created. This includes `embedding_model` and any RAG-specific build parameters
such as chunking, sparse model, graph extraction model, or preparation controls.
Changing those values requires building a new index.

Evaluation and playground requests may provide query-time overrides. Supported
top-level overrides are:

- `llm_model`: RAG generation or orchestration model.
- `top_k`: retrieval count passed to query execution.
- `parameters`: RAG-type-specific query-phase controls.

The DeepEval judge model is configured separately on evaluations through
`eval_judge_model`; it defaults to the effective RAG generation model when omitted.

## Common Local Profiles

### Simple Semantic Evaluation

```env
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-5-mini
EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=sqlite+aiosqlite:///./storage/dev.db
```

### Hybrid Search

```env
OPENAI_API_KEY=your_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=hybrid_rag
SPARSE_MODEL_NAME=prithvida/Splade_pp_en_v1
```

Start Qdrant:

```powershell
docker-compose up -d qdrant
```

### Graph RAG

```env
OPENAI_API_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

Start Neo4j:

```powershell
docker-compose up -d neo4j
```

### OpenAI-Compatible Provider

```env
OPENAI_API_KEY=your_provider_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openrouter/openai/gpt-5-mini
```

## Validation Commands

Core CLI settings:

```powershell
uv run python -c "from rag_evaluator.config import settings; print(settings.model_dump())"
```

Backend settings:

```powershell
cd platform/backend
uv run python -c "from app.config import settings; print(settings.model_dump())"
```

Backend health:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

## Secrets

Do not commit `.env`, API keys, database passwords, or downloaded datasets. For
production deployments, use your platform's secret manager and inject secrets as
environment variables.
