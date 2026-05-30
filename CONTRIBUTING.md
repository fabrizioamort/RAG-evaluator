# Contributing

Thanks for helping improve RAG Evaluator. This guide covers the development workflow,
quality checks, and main integration points.

## Project Areas

```text
src/rag_evaluator/        Core RAG implementations, evaluation engine, CLI
platform/backend/         FastAPI backend
platform/frontend/        React/Vite frontend
tests/                    Core test suite
docs/                     Public documentation
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- `uv`
- Docker and Docker Compose

## Setup

Core:

```powershell
uv sync --all-extras
Copy-Item .env.example .env
```

Backend:

```powershell
cd platform/backend
uv sync --all-extras
```

Frontend:

```powershell
cd platform/frontend
npm install
```

## Run Locally

From the repository root:

```powershell
docker-compose up -d postgres qdrant neo4j
```

Backend:

```powershell
cd platform/backend
uv run python dev_server.py
```

Frontend:

```powershell
cd platform/frontend
npm run dev
```

Open:

- Frontend: <http://localhost:3000>
- API docs: <http://localhost:8000/api/v1/docs>

## Quality Checks

Core:

```powershell
uv run pytest
uv run ruff check .
uv run mypy src/rag_evaluator
```

Backend:

```powershell
cd platform/backend
uv run pytest
uv run ruff check .
uv run mypy app
```

Frontend:

```powershell
cd platform/frontend
npm run lint
npm run build
```

Makefile shortcuts from the root:

```powershell
make test
make lint
make check
```

## Backend Model Changes

Any change under `platform/backend/app/models/` requires an Alembic migration.

```powershell
cd platform/backend
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
```

SQLite migrations that alter existing tables may need Alembic batch mode.

## Adding A RAG Implementation

See [Custom RAG Integration](docs/custom_rag_integration.md) for the full guide.

At a high level:

1. Add the implementation under `src/rag_evaluator/rag_implementations/`.
2. Implement `BaseRAG`.
3. Register the class and schema in `src/rag_evaluator/rag_implementations/registry.py`.
4. Add platform metadata in `platform/backend/app/services/rag_registry.py`.
5. Add parameter mapping in `platform/backend/app/services/rag_adapter.py`.
6. Add index storage mapping and cleanup in `index_build_service.py` if needed.
7. Add tests and documentation.

## Documentation

Update docs when behavior, commands, endpoints, configuration, or supported strategies
change. Keep `README.md` concise and link to detailed pages under `docs/`.

Before submitting documentation changes, scan for:

- Stale commands.
- Wrong ports.
- Placeholder comments.
- Broken links.
- Public claims that do not match the current code.

## Pull Requests

1. Create a focused branch.
2. Keep changes scoped to the issue or feature.
3. Add tests for behavior changes.
4. Update documentation for user-visible changes.
5. Run relevant quality checks.
6. Open a PR with a clear summary and testing notes.
