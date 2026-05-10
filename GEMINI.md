# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Project Overview

RAG Evaluator Platform is a comprehensive system for designing, testing, and evaluating RAG implementations. It consists of:

1. **Core CLI:** A Python-based command-line tool for local evaluations and RAG implementation logic.
2. **Platform (Web UI):** A production-ready web application (FastAPI + React) for managing projects and evaluations.

## Development Environment

- **OS:** Windows 11 (Powershell)
- **Package Manager:** `uv` (Python), `npm` (Frontend)
- **Infrastructure:** Docker & Docker Compose

### Setup Commands

```powershell
# Core/CLI Dependencies
uv sync --all-extras

# Backend Dependencies
cd platform/backend
uv sync

# Frontend Dependencies
cd platform/frontend
npm install
```

### Running the Application

**Platform (Recommended):**

```powershell
# Run Infrastructure (Databases)
docker-compose up -d postgres qdrant neo4j

# Run Backend
cd platform/backend
uv run python dev_server.py

# Run Frontend
cd platform/frontend
npm run dev
```

**CLI Tools:**

```powershell
# Prepare Data
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw

# Run Evaluation
uv run rag-eval evaluate --rag-type vector_semantic

# Launch Legacy UI
uv run rag-eval ui
```

## Architecture

### Directory Structure

- `src/rag_evaluator/`: **Core Logic** (RAG implementations, DeepEval integration, CLI).
- `platform/backend/`: **API** (FastAPI, SQLModel, PostgreSQL/SQLite).
- `platform/frontend/`: **UI** (React, Vite, Tailwind).
- `data/`: **Storage** (Raw docs, indexes, reports).

### Core Design Pattern (RAG)

- `BaseRAG` (`src/rag_evaluator/common/base_rag.py`) defines the interface.
- Implementations:
  - `vector_semantic`: ChromaDB + OpenAI.
  - `vector_hybrid`: Qdrant (Dense + Sparse/SPLADE) + RRF.
  - `graph_rag`: Neo4j + GraphRAG.
  - `filesystem_rag`: Agentic file retrieval.

## Testing & Quality

**Core:**

```powershell
uv run pytest
uv run ruff check .
uv run mypy src/rag_evaluator
```

**Backend:**

```powershell
cd platform/backend
uv run pytest
uv run ruff check .
```

**Frontend:**

```powershell
cd platform/frontend
npm run lint
```

## Configuration

- **Global:** `.env` file in root (shared by Core and Docker).
- **Backend:** Uses `platform/backend/.env` or falls back to environment variables.
- **Key Variables:**
  - `OPENAI_API_KEY`: Required.
  - `DATABASE_URL`: Connection string.
  - `QDRANT_URL`: Vector DB URL.

## Documentation Map

- `README.md`: High-level overview and quick start.
- `docs/cli.md`: Detailed CLI usage and RAG implementation guides.
- `docs/api.md`: Backend API endpoints.
- `docs/deployment.md`: Docker/Production setup.
- `CONTRIBUTING.md`: Developer guide.

## AI Collaboration Rules

- **Execution Context**: Always `cd` into the relevant subdirectory (`platform/backend` or `platform/frontend`) before running tools. Shell commands run from the root often fail to find the correct `.venv`.
- **Database Changes**: Any modification to `app/models/` requires a new Alembic migration. SQLite requires "batch mode" migrations for most operations.
- **Testing**: Run tests from the `platform/backend` directory. If `aiosqlite` is missing, the environment context is likely incorrect.
- **UI Edits**: When editing React components, prioritize maintaining the existing JSX nesting. Verify closing tags after every block insertion.
