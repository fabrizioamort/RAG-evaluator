# Getting Started

This guide gets you from a fresh clone to a working evaluation. The examples use
PowerShell because the project is developed primarily on Windows, but the same commands
map directly to Bash with `cp` instead of `Copy-Item`.

## Prerequisites

| Tool | Version |
| --- | --- |
| Python | 3.11 or newer |
| Node.js | 18 or newer |
| `uv` | Current stable release |
| Docker | 24 or newer |
| Docker Compose | v2 |

You also need an API key for at least one LLM provider. OpenAI is the default provider.

## Local Web Platform

This setup runs infrastructure in Docker and runs the backend/frontend directly from
the repository. It is the recommended development workflow.

### 1. Clone And Configure

```powershell
git clone https://github.com/fabrizioamort/RAG-evaluator.git
cd RAG-evaluator
Copy-Item .env.example .env
```

Edit `.env` and set:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5-mini
EMBEDDING_MODEL=text-embedding-3-small
```

If you use an OpenAI-compatible provider, also set:

```env
OPENAI_BASE_URL=https://your-provider.example/v1
```

### 2. Install Dependencies

```powershell
uv sync --all-extras

cd platform/backend
uv sync --all-extras

cd ..\frontend
npm install
```

### 3. Start Infrastructure

From the repository root:

```powershell
docker-compose up -d postgres qdrant neo4j
```

The root `docker-compose.yml` starts infrastructure only:

- PostgreSQL on `localhost:5432`
- Qdrant on `localhost:6333`
- Neo4j browser on `localhost:7474`, Bolt on `localhost:7687`

### 4. Start Backend

In a new terminal:

```powershell
cd platform/backend
uv run python dev_server.py
```

The backend is available at:

- API: <http://localhost:8000>
- OpenAPI docs: <http://localhost:8000/api/v1/docs>

The development launcher can clean up a stale Windows listener on port 8000 before
starting the server. You can also run cleanup only:

```powershell
uv run python dev_server.py --kill-port 8000
```

### 5. Start Frontend

In another terminal:

```powershell
cd platform/frontend
npm run dev
```

Open <http://localhost:3000>.

## First Web Evaluation

### 1. Create A Project

Open the web UI, create a project, and give it a descriptive name such as
`Support Docs Evaluation`.

### 2. Create A Knowledge Base

Create a knowledge base in the project and upload your documents. Start with a small
set of PDF, DOCX, TXT, or Markdown files while validating the workflow.

### 3. Create A RAG Configuration

Create a RAG configuration. Start with:

```text
RAG type: vector_semantic
Provider: openai
Model: gpt-5-mini
```

Leave storage parameters blank unless you need a custom external collection or path.

### 4. Build An Index

Create an index from the knowledge base and RAG configuration. Wait until the index
status is `ready`.

For other RAG types:

- `vector_hybrid` requires Qdrant.
- `graph_rag` requires Neo4j.
- `filesystem_rag` and `rlm_rag` use local prepared filesystem storage.

### 5. Create Or Import A Test Set

You can create test cases manually, import JSON, or generate cases from a knowledge
base and review them.

Import JSON shape:

```json
{
  "name": "Smoke tests",
  "description": "Initial questions for the support docs",
  "tags": ["smoke"],
  "test_cases": [
    {
      "question": "How does a user reset a password?",
      "expected_answer": "Users reset passwords from account settings.",
      "ground_truth_context": [
        "Password reset instructions are available in account settings."
      ],
      "difficulty": "easy",
      "question_type": "factual"
    }
  ]
}
```

### 6. Run An Evaluation

Start an evaluation from the project. Select:

- A ready knowledge base index.
- A test set.
- Metrics to calculate.

For first runs, use `faithfulness` and `g_eval` to control cost. Add contextual
precision and recall once the pipeline is stable.

### 7. Inspect Results

Review:

- Summary metrics and pass rate.
- Per-question generated answer and expected answer.
- LLM judge reasoning.
- Retrieval trace and retrieved chunks.
- Latency, token usage, and estimated cost.
- Run manifest for reproducibility.

Mark a completed evaluation as baseline when it represents the current reference
system.

### 8. Compare Alternatives

Build a second index using another RAG configuration, run it against the same test set,
then create a comparison in the project comparison tab. Compare aggregate deltas,
per-question differences, cost, and latency.

## CLI Workflow

The CLI is useful for quick local experiments and CI-style report generation.

### 1. Prepare Documents

```powershell
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw
```

### 2. Create A Test Set

Save `data/test_set.json`:

```json
{
  "test_cases": [
    {
      "question": "What is RAG?",
      "expected_answer": "RAG combines retrieval with generation.",
      "ground_truth_context": [
        "Retrieval Augmented Generation combines retrieval with generation."
      ],
      "difficulty": "easy"
    }
  ]
}
```

### 3. Evaluate

```powershell
uv run rag-eval evaluate --rag-type vector_semantic --test-set data/test_set.json --verbose
```

Reports are written to `reports/`.

### 4. Compare Multiple RAG Types

```powershell
uv run rag-eval evaluate --rag-type all --test-set data/test_set.json
```

You can also evaluate one implementation and combine it with previous report files:

```powershell
uv run rag-eval evaluate --rag-type vector_hybrid --combine
```

## Full Docker Stack

The full application stack is defined under `docker/`.

```powershell
cd docker
docker compose up -d
```

Enable optional infrastructure profiles when needed:

```powershell
docker compose --profile hybrid --profile graph up -d
```

See [Deployment](../deployment.md) for production notes.

## Next Steps

- Read [RAG Strategies](../rag_strategies.md) before selecting a non-baseline retriever.
- Read [Metrics](../metrics.md) before deciding which metrics to run at scale.
- Read [Configuration](configuration.md) before changing provider, database, or storage settings.
- Read [Troubleshooting](troubleshooting.md) if services fail to start or evaluations return empty results.
