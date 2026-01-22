# Configuration Reference

> **Complete reference for all configuration options in the RAG Evaluator Platform.**

The platform is configured primarily through environment variables, stored in a `.env` file at the project root.

---

## Table of Contents

- [Quick Setup](#quick-setup)
- [LLM Configuration](#llm-configuration)
- [Database Configuration](#database-configuration)
- [Vector Store Configuration](#vector-store-configuration)
- [Evaluation Configuration](#evaluation-configuration)
- [Performance Tuning](#performance-tuning)
- [Storage Configuration](#storage-configuration)
- [Logging Configuration](#logging-configuration)
- [Configuration by Environment](#configuration-by-environment)

---

## Quick Setup

Start with the example file:

```bash
cp .env.example .env
```

**Minimum required configuration:**

```env
# Required - Your OpenAI API key
OPENAI_API_KEY=sk-your-api-key-here
```

All other settings have sensible defaults.

---

## LLM Configuration

### OpenAI

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | **Required** |
| `OPENAI_MODEL` | Model for generation | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Model for embeddings | `text-embedding-3-small` |
| `OPENAI_TIMEOUT` | API timeout (seconds) | `600` |
| `OPENAI_TEMPERATURE` | Generation temperature | `0.0` |
| `OPENAI_MAX_TOKENS` | Max tokens per response | `1000` |

**Example configurations:**

```env
# Cost-optimized (default)
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Quality-optimized
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large

# Budget-constrained
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
```

### Alternative Providers (via LiteLLM)

The platform supports other LLM providers through LiteLLM:

```env
# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key
OPENAI_MODEL=claude-3-sonnet-20240229

# Azure OpenAI
AZURE_API_KEY=your-azure-key
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
OPENAI_MODEL=azure/your-deployment-name

# Local Ollama
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_MODEL=ollama/llama2
```

---

## Database Configuration

### PostgreSQL (Recommended for Production)

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Full connection string | SQLite fallback |
| `DB_PASSWORD` | Database password | - |
| `DB_HOST` | Database host | `postgres` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `rag_eval` |
| `DB_USER` | Database user | `postgres` |

**Connection string format:**

```env
DATABASE_URL=postgresql+asyncpg://user:password@host:port/database

# Docker Compose
DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/rag_eval

# Local PostgreSQL
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/rag_eval
```

### SQLite (Development)

```env
# Uses SQLite automatically if DATABASE_URL is not set
DATABASE_URL=sqlite+aiosqlite:///./data/rag_eval.db
```

### Connection Pool Settings

```env
# For high-concurrency scenarios
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
```

---

## Vector Store Configuration

### ChromaDB (Vector Semantic)

| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMA_PERSIST_DIRECTORY` | Storage path | `./data/chroma_db` |
| `CHROMA_COLLECTION_NAME` | Collection name | `rag_documents` |

```env
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
```

### Qdrant (Hybrid Search)

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_COLLECTION_NAME` | Collection name | `hybrid_rag` |
| `QDRANT_API_KEY` | API key (if secured) | - |

```env
# Local development
QDRANT_URL=http://localhost:6333

# Docker Compose
QDRANT_URL=http://qdrant:6333

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

### Neo4j (Graph RAG)

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Username | `neo4j` |
| `NEO4J_PASSWORD` | Password | - |

```env
# Local development
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Docker Compose
NEO4J_URI=bolt://neo4j:7687

# Neo4j AuraDB
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
```

---

## Evaluation Configuration

### Metric Thresholds

| Variable | Description | Default |
|----------|-------------|---------|
| `EVAL_FAITHFULNESS_THRESHOLD` | Pass threshold for Faithfulness | `0.7` |
| `EVAL_ANSWER_RELEVANCY_THRESHOLD` | Pass threshold for Answer Relevancy | `0.7` |
| `EVAL_CONTEXTUAL_PRECISION_THRESHOLD` | Pass threshold for Precision | `0.7` |
| `EVAL_CONTEXTUAL_RECALL_THRESHOLD` | Pass threshold for Recall | `0.7` |

```env
# Strict thresholds for production
EVAL_FAITHFULNESS_THRESHOLD=0.85
EVAL_ANSWER_RELEVANCY_THRESHOLD=0.80
EVAL_CONTEXTUAL_PRECISION_THRESHOLD=0.75
EVAL_CONTEXTUAL_RECALL_THRESHOLD=0.80

# Lenient thresholds for development
EVAL_FAITHFULNESS_THRESHOLD=0.6
EVAL_ANSWER_RELEVANCY_THRESHOLD=0.6
```

### DeepEval Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPEVAL_ASYNC_MODE` | Enable parallel evaluation | `False` |
| `DEEPEVAL_MAX_CONCURRENT` | Max concurrent evaluations | `10` |
| `DEEPEVAL_THROTTLE_VALUE` | Delay between requests (seconds) | `0.0` |
| `DEEPEVAL_PER_TASK_TIMEOUT` | Total timeout per test case | `900` |
| `DEEPEVAL_PER_ATTEMPT_TIMEOUT` | Timeout per API call | `300` |
| `DEEPEVAL_MAX_RETRIES` | Max retry attempts | `3` |

```env
# Fast evaluation (requires high API limits)
DEEPEVAL_ASYNC_MODE=True
DEEPEVAL_MAX_CONCURRENT=10
DEEPEVAL_THROTTLE_VALUE=0.0

# Rate-limited (conservative)
DEEPEVAL_ASYNC_MODE=False
DEEPEVAL_MAX_CONCURRENT=3
DEEPEVAL_THROTTLE_VALUE=1.0
```

### Test Set Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `EVAL_TEST_SET_PATH` | Default test set file | `data/test_set.json` |
| `EVAL_REPORTS_DIR` | Report output directory | `reports` |

---

## Performance Tuning

### Chunking Configuration

| Variable | Description | Default | Impact |
|----------|-------------|---------|--------|
| `HYBRID_CHUNK_SIZE` | Characters per chunk | `700` | Larger = more context, slower |
| `HYBRID_CHUNK_OVERLAP` | Overlap between chunks | `100` | Larger = better continuity |
| `HYBRID_INDEXING_BATCH_SIZE` | Batch size for indexing | `16` | Larger = faster, more memory |
| `HYBRID_FUSION_ALPHA` | Dense vs. sparse weight | `0.5` | Higher = more semantic |

```env
# For short documents
HYBRID_CHUNK_SIZE=500
HYBRID_CHUNK_OVERLAP=50

# For long documents with complex reasoning
HYBRID_CHUNK_SIZE=1000
HYBRID_CHUNK_OVERLAP=200
```

### Sparse Embeddings (SPLADE)

| Variable | Description | Default |
|----------|-------------|---------|
| `SPARSE_MODEL_NAME` | SPLADE model | `prithvida/Splade_pp_en_v1` |

```env
# Alternative models
SPARSE_MODEL_NAME=naver/splade-cocondenser-ensembledistil
SPARSE_MODEL_NAME=naver/splade-cocondenser-selfdistil
```

### Filesystem RAG

| Variable | Description | Default |
|----------|-------------|---------|
| `FILESYSTEM_RAG_MAX_ITERATIONS` | Max ReAct iterations | `10` |
| `FILESYSTEM_RAG_MAX_TOOL_CALLS` | Max tool calls | `20` |
| `FILESYSTEM_RAG_MAX_FILE_READS` | Max file reads | `10` |
| `FILESYSTEM_RAG_WORD_THRESHOLD` | LLM vs. heuristic threshold | `1000` |

---

## Storage Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `STORAGE_PATH` | Base storage directory | `./storage` |
| `RAW_DATA_DIR` | Raw document input | `data/raw` |
| `PROCESSED_DATA_DIR` | Processed output | `data/processed` |
| `UPLOADS_DIR` | Uploaded files | `storage/uploads` |
| `INDEXES_DIR` | Built indexes | `storage/indexes` |

```env
# Custom storage paths
STORAGE_PATH=/var/lib/rag-evaluator/storage
RAW_DATA_DIR=/data/documents/raw
```

### Storage Structure

```
storage/
├── uploads/           # Uploaded documents
│   └── {kb_id}/
│       └── documents/
├── indexes/           # Built indexes
│   └── {index_id}/
│       ├── chroma/    # ChromaDB data
│       ├── qdrant/    # Qdrant data (if local)
│       └── filesystem_rag/
└── artifacts/         # Generated artifacts
```

---

## Logging Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `LITELLM_LOGGING` | LiteLLM detailed logs | `false` |
| `LOG_FORMAT` | Log format | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` |

```env
# Development (verbose)
LOG_LEVEL=DEBUG
LITELLM_LOGGING=true

# Production (quiet)
LOG_LEVEL=WARNING
LITELLM_LOGGING=false
```

---

## Configuration by Environment

### Development

```env
# .env.development

# LLM - Cost-optimized
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Database - SQLite for simplicity
DATABASE_URL=sqlite+aiosqlite:///./data/rag_eval.db

# Vector Stores - Local
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=dev_password

# Evaluation - Fast, lenient
DEEPEVAL_ASYNC_MODE=True
DEEPEVAL_MAX_CONCURRENT=5
EVAL_FAITHFULNESS_THRESHOLD=0.6

# Logging - Verbose
LOG_LEVEL=DEBUG
LITELLM_LOGGING=true
```

### Production

```env
# .env.production

# LLM - Quality-optimized
OPENAI_API_KEY=sk-your-production-key
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
OPENAI_TEMPERATURE=0

# Database - PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:password@db.example.com:5432/rag_eval
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# Vector Stores - Managed services
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=production-key
NEO4J_URI=neo4j+s://production.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=secure_password

# Evaluation - Strict, rate-limited
DEEPEVAL_ASYNC_MODE=True
DEEPEVAL_MAX_CONCURRENT=10
DEEPEVAL_THROTTLE_VALUE=0.5
EVAL_FAITHFULNESS_THRESHOLD=0.85

# Timeouts - Generous for reliability
OPENAI_TIMEOUT=900
DEEPEVAL_PER_TASK_TIMEOUT=1800

# Logging - Production
LOG_LEVEL=INFO
LITELLM_LOGGING=false
```

### Docker Compose

```env
# .env (for docker-compose)

OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini

# Use Docker service names
DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/rag_eval
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

DEEPEVAL_ASYNC_MODE=True
LOG_LEVEL=INFO
```

---

## Configuration Validation

The platform validates configuration on startup. Common validation errors:

| Error | Cause | Solution |
|-------|-------|----------|
| `OPENAI_API_KEY not set` | Missing API key | Add key to `.env` |
| `Invalid DATABASE_URL` | Malformed connection string | Check format |
| `Cannot connect to Qdrant` | Qdrant not running | Start Qdrant service |
| `Neo4j authentication failed` | Wrong credentials | Verify username/password |

### Manual Validation

```bash
# Check config loading
cd platform/backend
uv run python -c "from app.config import settings; print(settings.dict())"

# Test database connection
uv run python -c "from app.database import engine; print('DB OK')"

# Test OpenAI connection
uv run python -c "
from openai import OpenAI
client = OpenAI()
print(client.models.list().data[0].id)
"
```

---

## Environment Variable Precedence

Configuration is loaded in this order (later overrides earlier):

1. **Defaults** - Hardcoded in `config.py`
2. **`.env` file** - Project root
3. **Environment variables** - System/shell environment
4. **CLI arguments** - Command-line flags (where applicable)

```bash
# Override with environment variable
OPENAI_MODEL=gpt-4o uv run rag-eval evaluate --rag-type vector_semantic

# Override for Docker
docker-compose run -e OPENAI_MODEL=gpt-4o backend python -m app.main
```

---

## Secrets Management

**Never commit secrets to version control.**

### Best Practices

1. **Use `.env.example`** for documentation (no real values)
2. **Add `.env` to `.gitignore`**
3. **Use different keys** for development/production
4. **Rotate keys regularly**

### Production Options

| Method | Best For |
|--------|----------|
| Environment variables | Container orchestration (K8s, ECS) |
| Secret manager | Cloud deployments (AWS Secrets, GCP Secret Manager) |
| Vault | Enterprise, multi-environment |

```yaml
# Kubernetes example
env:
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: rag-evaluator-secrets
        key: openai-api-key
```

---

## Related Documentation

- [Deployment Guide](../deployment.md) - Production deployment
- [Troubleshooting Guide](troubleshooting.md) - Configuration issues
- [Architecture Overview](../ARCHITECTURE.md) - System design
