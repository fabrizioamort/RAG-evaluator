# RAG Evaluator Backend

FastAPI backend for the RAG Evaluator Platform.

## Responsibilities

- Project, knowledge base, test set, RAG config, index, evaluation, comparison,
  trend, playground, template, and webhook APIs.
- Async database access through SQLAlchemy.
- Background index builds and evaluations.
- Server-Sent Events for index/evaluation progress.
- Artifact storage for retrieval traces, contexts, generated-test provenance, and raw metrics.

## Setup

Run backend commands from `platform/backend`.

```powershell
cd platform/backend
uv sync --all-extras
Copy-Item .env.example .env
```

For local SQLite, the default is enough:

```env
DATABASE_URL=sqlite+aiosqlite:///./storage/dev.db
STORAGE_PATH=./storage
```

For PostgreSQL, start infrastructure from the repository root:

```powershell
docker-compose up -d postgres
```

Then set:

```env
DATABASE_URL=postgresql+asyncpg://rageval:rageval@localhost:5432/rageval
```

## Run

```powershell
uv run python dev_server.py
```

The development launcher starts Uvicorn with reload on port 8000. On Windows it can
clean up stale listeners:

```powershell
uv run python dev_server.py --kill-port 8000
```

Open:

- API root: <http://localhost:8000>
- OpenAPI docs: <http://localhost:8000/api/v1/docs>
- Health: <http://localhost:8000/api/v1/health>

## Tests And Quality

```powershell
uv run pytest
uv run ruff check .
uv run mypy app
```

Run focused tests:

```powershell
uv run pytest tests/test_api/test_health.py -q
uv run pytest tests/test_services/test_rag_adapter.py -q
```

## Migrations

```powershell
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "describe change"
uv run alembic downgrade -1
```

Any model change under `app/models/` requires an Alembic migration. SQLite schema
changes may require Alembic batch mode.

## Configuration

Important variables:

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite+aiosqlite:///./storage/dev.db` | Backend database URL. |
| `STORAGE_PATH` | `./storage` | Documents, indexes, artifacts, reports, logs. |
| `LOG_LEVEL` | `INFO` | Backend log level. |
| `LOG_FORMAT` | `json` | `json` or `console`. |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed frontend origins. |
| `OPENAI_API_KEY` | unset | OpenAI key. |
| `OPENROUTER_API_KEY` | unset | OpenRouter key. |
| `ANTHROPIC_API_KEY` | unset | Anthropic key. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama URL. |
| `DEFAULT_LLM_PROVIDER` | `openai` | Default provider metadata. |
| `DEFAULT_LLM_MODEL` | `gpt-4o-mini` | Default backend model. |

See [Configuration](../../docs/guides/configuration.md) for the full reference.

## Structure

```text
platform/backend/
  app/
    api/          FastAPI routers
    models/       SQLAlchemy models
    schemas/      Pydantic schemas
    services/     Business logic
    utils/        Logging, errors, templates
    config.py     Settings
    database.py   Async database engine/session
    main.py       FastAPI app
  alembic/        Migrations
  tests/          Backend tests
```
