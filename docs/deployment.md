# Deployment Guide - RAG Evaluation Platform

This guide covers the deployment of the RAG Evaluation Platform using Docker and Docker Compose.

## Prerequisites

- **Docker** 24.0 or higher
- **Docker Compose** v2.0 or higher
- **OpenAI API Key** (or other supported providers like Ollama, Anthropic)

## Quick Start (Pre-built Images)

For the simplest setup, you can use the provided `docker-compose.yml` file to start all services.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/RAG-evaluator.git
   cd RAG-evaluator
   ```

2. **Configure environment variables:**
   Copy the example environment file and edit it with your settings.

   ```bash
   cp .env.example .env
   ```

   At a minimum, ensure `OPENAI_API_KEY` is set.

3. **Start the platform:**

   ```bash
   docker-compose up -d
   ```

4. **Access the UI:**
   Open your browser and navigate to `http://localhost:3000`.

## Configuration

The platform is configured via environment variables in the `.env` file.

| Variable | Description | Default |
| --- | --- | --- |
| `DATABASE_URL` | Connection string for PostgreSQL or SQLite | `sqlite+aiosqlite:///./data/rag_eval.db` |
| `STORAGE_PATH` | Directory for documents and indexes | `./storage` |
| `LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `OPENAI_API_KEY` | Your OpenAI API key | None |
| `LITELLM_LOGGING` | Enable detailed LiteLLM logs | `false` |

## Production Deployment

### Using PostgreSQL

For production, it is recommended to use PostgreSQL instead of SQLite.

1. Update `DATABASE_URL` in your `.env`:

   ```env
   DATABASE_URL=postgresql+asyncpg://user:password@db:5432/rag_eval
   ```

2. The default `docker-compose.yml` includes a PostgreSQL service. Ensure the credentials match.

### Storage and Persistence

The platform stores uploaded documents and search indexes in volumes. Ensure these volumes are backed up regularly.

- `rag_eval_data`: Stores the PostgreSQL data.
- `rag_eval_storage`: Stores uploaded files, artifacts, and indexes.

## Troubleshooting

### Logs

To view logs for all services:

```bash
docker-compose logs -f
```

To view logs for a specific service:

```bash
docker-compose logs -f backend
```

### Resetting the Database

To perform a clean reset of the database and storage:

```bash
docker-compose down -v
rm -rf storage/*
docker-compose up -d
```
