# Troubleshooting

Use this guide when setup, indexing, or evaluations fail.

## Quick Checks

From the repository root:

```powershell
docker-compose ps
Invoke-RestMethod http://localhost:6333/health
```

Backend:

```powershell
cd platform/backend
uv run python -c "from app.config import settings; print(settings.model_dump())"
uv run pytest tests/test_api/test_health.py -q
```

Frontend:

```powershell
cd platform/frontend
npm run build
```

CLI:

```powershell
uv run rag-eval --help
uv run python -c "from rag_evaluator.config import settings; print(settings.model_dump())"
```

## Backend Will Not Start

### Port 8000 Is Already In Use

The backend launcher can stop stale listeners on Windows:

```powershell
cd platform/backend
uv run python dev_server.py --kill-port 8000
uv run python dev_server.py
```

If access is denied, the process may be elevated or owned by another environment. Stop
that process manually or run from an Administrator PowerShell.

### Wrong Virtual Environment

Backend commands must run from `platform/backend`:

```powershell
cd platform/backend
uv sync --all-extras
uv run python dev_server.py
```

If tests report missing `aiosqlite`, you are probably running from the wrong directory.

### Database Connection Fails

For local SQLite:

```env
DATABASE_URL=sqlite+aiosqlite:///./storage/dev.db
```

For local PostgreSQL, start the service:

```powershell
docker-compose up -d postgres
```

Then set a matching URL:

```env
DATABASE_URL=postgresql+asyncpg://rageval:rageval@localhost:5432/rageval
```

## Frontend Issues

### Blank Page

1. Check the browser console.
2. Verify the backend is running:

   ```powershell
   Invoke-RestMethod http://localhost:8000/api/v1/health
   ```

3. Rebuild the frontend:

   ```powershell
   cd platform/frontend
   npm run build
   ```

### API Calls Fail In Development

The Vite dev server proxies `/api` to `http://localhost:8000`. Confirm the backend is
on port 8000 and the frontend was started with:

```powershell
cd platform/frontend
npm run dev
```

## Provider And Model Issues

### OpenAI Authentication Fails

Check `.env`:

```powershell
Get-Content .env | Select-String OPENAI_API_KEY
```

Make sure the value has no quotes, trailing spaces, or placeholder text. Restart the
backend or rerun the CLI after changing `.env`.

### OpenAI-Compatible Provider Does Not Work

Set both key and base URL:

```env
OPENAI_API_KEY=your_provider_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

For platform RAG configs, also set the provider/model in the UI. If the provider uses
OpenAI-compatible routing, store the base URL in the config where available.

### Rate Limits Or Timeouts

Use conservative settings:

```env
DEEPEVAL_ASYNC_MODE=False
DEEPEVAL_MAX_CONCURRENT=3
DEEPEVAL_THROTTLE_VALUE=1.0
OPENAI_TIMEOUT=900
DEEPEVAL_PER_TASK_TIMEOUT=1800
DEEPEVAL_PER_ATTEMPT_TIMEOUT=600
```

Reduce the test set size while debugging.

## Indexing Issues

### Qdrant Connection Refused

Hybrid search requires Qdrant:

```powershell
docker-compose up -d qdrant
Invoke-RestMethod http://localhost:6333/health
```

Set:

```env
QDRANT_URL=http://localhost:6333
```

### SPLADE Or FastEmbed Model Download Fails

Hybrid search downloads the configured sparse model on first use. Check network access
and the `SPARSE_MODEL_NAME` value. Retry with a smaller document subset after the model
is cached.

### Neo4j Connection Fails

Graph RAG requires Neo4j:

```powershell
docker-compose up -d neo4j
```

Wait for startup, then open <http://localhost:7474>. Confirm:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

Graph RAG uses LLM calls during graph construction. Start with a small subset to reduce
cost and isolate failures.

### Filesystem Or RLM-RAG Preparation Fails

Check that the input directory exists and contains supported files:

```powershell
Get-ChildItem data/raw
```

For platform builds, inspect the index error message in the UI and backend logs. For CLI
runs, rerun with a small directory and confirm the prepared output path is writable.

## Evaluation Issues

### Evaluation Stays Pending Or Running

Check backend logs and health:

```powershell
cd platform/backend
uv run pytest tests/test_api/test_health.py -q
```

Confirm the selected index is `ready`, the test set has cases, and the provider key is
valid.

### Results Are Empty Or All Scores Are Zero

Likely causes:

- The index was not built successfully.
- The RAG returned no context.
- The test set lacks `expected_answer`.
- Provider calls failed during generation or judging.

Use the playground to query the same index. If the playground returns no useful context,
debug indexing and retrieval before rerunning metrics.

### CLI Says Test Set Has Zero Cases

The CLI expects:

```json
{
  "test_cases": []
}
```

A raw top-level JSON array is not accepted by the current CLI evaluator.

### Pause Or Resume Does Not Behave As Expected

Pause/resume uses backend checkpointing and event logs. It applies to platform
evaluations, not CLI runs. If resume fails, retry the evaluation from the UI.

## Import Issues

### JSON Test Set Import Fails

The platform import endpoint expects:

```json
{
  "name": "Imported test set",
  "test_cases": [
    {
      "question": "Question",
      "expected_answer": "Answer"
    }
  ]
}
```

Optional fields include `description`, `tags`, `ground_truth_context`, `difficulty`,
`category`, and `question_type`.

## Storage And Cleanup

Backend storage lives under `STORAGE_PATH`, usually `platform/backend/storage` for
backend-local SQLite development or repository `storage/` for full-stack/container
usage depending on how you launched the app.

Deleting an index removes associated physical storage when the backend knows how to
clean up that storage type. Archive indexes if evaluations still need to reference
them.

For a clean local infrastructure reset:

```powershell
docker-compose down -v
```

For full stack:

```powershell
cd docker
docker compose down -v
```

This deletes container volumes.

## Getting Help

When opening an issue, include:

- Operating system.
- Python, Node, `uv`, and Docker versions.
- Exact command that failed.
- Sanitized `.env` values.
- Backend logs or CLI traceback.
- Whether you used root infrastructure Compose or the full stack under `docker/`.
