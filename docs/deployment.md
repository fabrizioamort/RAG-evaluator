# Deployment

This guide covers local infrastructure, the containerized application stack, and
production considerations.

## Compose Files

The repository has two Compose entry points with different purposes.

| File | Purpose |
| --- | --- |
| `docker-compose.yml` | Root development infrastructure only: PostgreSQL, Qdrant, Neo4j. |
| `docker/docker-compose.yml` | Full application stack: PostgreSQL, backend, frontend, optional Qdrant and Neo4j profiles. |
| `docker/docker-compose.dev.yml` | Containerized backend/frontend using SQLite for development. |

## Development Infrastructure

From the repository root:

```powershell
docker-compose up -d postgres qdrant neo4j
```

This starts:

| Service | Port | Notes |
| --- | --- | --- |
| PostgreSQL | `5432` | Metadata database for the platform. |
| Qdrant | `6333`, `6334` | Required for `vector_hybrid`. |
| Neo4j | `7474`, `7687` | Required for `graph_rag`. |

Use this mode when running backend and frontend locally:

```powershell
cd platform/backend
uv run python dev_server.py

cd platform/frontend
npm run dev
```

## Full Docker Stack

The full stack Compose file lives in `docker/`.

```powershell
cd docker
docker compose up -d
```

Default services:

- PostgreSQL database.
- FastAPI backend on `localhost:8000`.
- Frontend served on `localhost:3000`.

Optional profiles:

```powershell
# Include Qdrant for hybrid search
docker compose --profile hybrid up -d

# Include Neo4j for graph RAG
docker compose --profile graph up -d

# Include both
docker compose --profile hybrid --profile graph up -d
```

The full stack reads environment values from your shell or `.env` file. At minimum set:

```env
OPENAI_API_KEY=your_openai_key
DB_PASSWORD=change_me
```

## Containerized SQLite Development

For a simple containerized app without PostgreSQL:

```powershell
cd docker
docker compose -f docker-compose.dev.yml up -d
```

This runs backend and frontend containers with SQLite storage mounted from the repository
`storage/` directory.

## Production Checklist

Before exposing the platform beyond local development:

- Put the app behind TLS.
- Add authentication at a reverse proxy, VPN, private network, or API gateway.
- Use PostgreSQL rather than SQLite.
- Store API keys and database passwords in a secret manager.
- Back up PostgreSQL and `storage/`.
- Restrict database, Qdrant, and Neo4j ports to trusted networks.
- Set `LOG_FORMAT=json` for structured logs.
- Configure resource limits for backend, Qdrant, and Neo4j.
- Review uploaded document sensitivity before indexing.

## Environment

Common production variables:

```env
DATABASE_URL=postgresql+asyncpg://rageval:strong_password@db:5432/rageval
STORAGE_PATH=/app/storage
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-5-mini
EMBEDDING_MODEL=text-embedding-3-small
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_AUTH=neo4j/strong_password
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=strong_password
LOG_LEVEL=INFO
LOG_FORMAT=json
CORS_ORIGINS=["https://rag-eval.example.com"]
```

See [Configuration](guides/configuration.md) for the full reference.

## Google Cloud (Vertex AI Gemini)

Vertex AI Gemini uses Application Default Credentials (ADC) — no API keys are
stored in the platform. Auth resolves in this order:

1. `GOOGLE_APPLICATION_CREDENTIALS` (service-account JSON path), if set.
2. User credentials from `gcloud auth application-default login` (dev only).
3. The metadata-server credentials of the attached service account (Cloud Run,
   GKE, Compute Engine).

Prefer option 3 in production. Grant the runtime service account:

- `roles/aiplatform.user` — required to call Gemini generation/embedding.
- `roles/discoveryengine.viewer` — required only when using
  `google_vertex_search` as a RAG retriever.

Environment variables:

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project
GOOGLE_CLOUD_LOCATION=us-central1        # us-central1, europe-west4, ...
VERTEX_GEMINI_MODEL=gemini-2.5-pro
VERTEX_GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

The core CLI talks to
`https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/endpoints/openapi`;
the platform backend uses LiteLLM's native `vertex_ai/*` path (same ADC).

## Database Migrations

Run backend migration commands from `platform/backend`:

```powershell
cd platform/backend
uv run alembic upgrade head
```

Create a migration after model changes:

```powershell
uv run alembic revision --autogenerate -m "describe change"
```

SQLite has limited schema-alteration support, so migrations that alter existing tables
may need Alembic batch mode.

## Persistence And Backups

Back up:

- PostgreSQL database.
- `storage/documents`.
- `storage/indexes`.
- `storage/artifacts`.
- `storage/logs/jobs` if you need job event history.
- Qdrant and Neo4j volumes when using local containers.

For a clean local reset:

```powershell
docker-compose down -v
```

For the full stack:

```powershell
cd docker
docker compose down -v
```

Deleting volumes removes local database and vector/graph storage.

## Logs

Root infrastructure:

```powershell
docker-compose logs -f postgres
docker-compose logs -f qdrant
docker-compose logs -f neo4j
```

Full stack:

```powershell
cd docker
docker compose logs -f backend
docker compose logs -f frontend
```

## Health Checks

Backend:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

Qdrant:

```powershell
Invoke-RestMethod http://localhost:6333/health
```

Neo4j:

Open <http://localhost:7474> or use `cypher-shell` inside the container.

## Security

The open source platform is unauthenticated by default. Treat it as local-only until
you add access control. See [Security](guides/security.md) for recommendations.
