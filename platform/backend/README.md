# RAG Evaluation Platform - Backend

FastAPI backend for the RAG Evaluation Platform.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- PostgreSQL 16 (production) or SQLite (development)

## Quick Start

### Development (SQLite)

```bash
# Install dependencies
cd platform/backend
uv sync --all-extras

# Create .env file
cp .env.example .env
# Edit .env with your API keys

# Run the development server
uv run python dev_server.py

# The API will be available at http://localhost:8000
# OpenAPI docs at http://localhost:8000/api/v1/docs
```

### Production (PostgreSQL + Docker)

```bash
# From the project root
cd docker
docker-compose up -d
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite+aiosqlite:///./storage/dev.db` |
| `STORAGE_PATH` | Path for file storage | `./storage` |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `LOG_FORMAT` | Log format (json/console) | `json` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `CORS_ORIGINS` | Allowed CORS origins (JSON array) | `["http://localhost:3000"]` |

## Database Migrations

```bash
# Run migrations
uv run alembic upgrade head

# Create a new migration
uv run alembic revision --autogenerate -m "Description"

# Downgrade
uv run alembic downgrade -1
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_api/test_health.py
```

## Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy app
```

## Project Structure

```
platform/backend/
├── app/
│   ├── api/           # API route handlers
│   ├── models/        # SQLAlchemy ORM models
│   ├── schemas/       # Pydantic schemas (DTOs)
│   ├── services/      # Business logic
│   ├── utils/         # Utilities (logging, etc.)
│   ├── config.py      # Application settings
│   ├── database.py    # Database configuration
│   └── main.py        # FastAPI application
├── alembic/           # Database migrations
├── tests/             # Test suite
└── pyproject.toml     # Project dependencies
```

## API Endpoints

### Health

- `GET /api/v1/health` - Health check
- `GET /api/v1/health/detail` - Detailed health info

### Projects (Coming in Phase 2)

- `GET /api/v1/projects` - List projects
- `POST /api/v1/projects` - Create project
- `GET /api/v1/projects/{id}` - Get project
- `PUT /api/v1/projects/{id}` - Update project
- `DELETE /api/v1/projects/{id}` - Delete project

See the OpenAPI documentation at `/api/v1/docs` for the complete API reference.
