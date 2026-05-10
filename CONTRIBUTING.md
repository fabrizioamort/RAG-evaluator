# Contributing to RAG Evaluator

Thank you for your interest in contributing to RAG Evaluator! This document provides guidelines and instructions for contributing to the project, whether you're working on the core RAG logic, the API backend, or the frontend UI.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Project Structure](#project-structure)
- [Development Environment](#development-environment)
- [Running the Platform Locally](#running-the-platform-locally)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Adding New RAG Implementations](#adding-new-rag-implementations)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project follows a simple principle: **Be respectful, be collaborative, and focus on improving the project.**

## Project Structure

The project is organized into three main areas:

```text
RAG-evaluator/
├── src/rag_evaluator/        # CORE: The shared RAG logic and CLI tool
│   ├── rag_implementations/  # The actual RAG strategies (Chroma, Hybrid, etc.)
│   ├── evaluation/           # DeepEval integration
│   └── ...
├── platform/                 # PLATFORM: The web application
│   ├── backend/              # FastAPI application
│   │   ├── app/              # API logic
│   │   └── ...
│   └── frontend/             # React/Vite application
│       ├── src/              # UI components and logic
│       └── ...
├── data/                     # Data storage (gitignored)
├── docker/                   # Docker configuration
└── tests/                    # Tests for the Core/CLI
```

## Development Environment

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for frontend)
- **[uv](https://github.com/astral-sh/uv)** (Python package manager)
- **Docker** (optional, but recommended for databases)

### 1. Core & CLI Setup

This is for working on `src/rag_evaluator` or the CLI.

```bash
# Install dependencies
uv sync --all-extras

# Set up environment
cp .env.example .env
```

### 2. Backend Setup

The backend is a FastAPI app located in `platform/backend`.

```bash
cd platform/backend

# Install backend dependencies
uv sync

# The backend shares the root .env, or you can create one in platform/backend/.env
```

### 3. Frontend Setup

The frontend is a React app located in `platform/frontend`.

```bash
cd platform/frontend

# Install dependencies
npm install
```

## Running the Platform Locally

For development, you often want to run services individually to get hot-reloading and debuggers.

### 1. Start Infrastructure (Databases)

Use Docker to start the required databases (PostgreSQL, Qdrant, Neo4j) without running the full app containers.

```bash
# From project root
docker-compose up -d postgres qdrant neo4j
```

### 2. Start Backend (API)

```bash
cd platform/backend
uv run python dev_server.py
```
API docs will be at: `http://localhost:8000/api/v1/docs`

### 3. Start Frontend (UI)

```bash
cd platform/frontend
npm run dev
```
The UI will be at: `http://localhost:5173`

## Code Quality Standards

We use strict linting and formatting.

### Python (Core & Backend)
- **Formatter:** `ruff format`
- **Linter:** `ruff check`
- **Type Checker:** `mypy`

```bash
# Run all checks for Core
make check

# Run all checks for Backend
cd platform/backend
uv run ruff check .
uv run mypy .
```

### TypeScript (Frontend)
- **Linter:** `eslint`

```bash
cd platform/frontend
npm run lint
```

## Testing Requirements

### Core/CLI Tests
Located in `tests/`.

```bash
# Run all core tests
uv run pytest
```

### Backend Tests
Located in `platform/backend/tests/`.

```bash
cd platform/backend
uv run pytest
```

### Frontend Tests
Located in `platform/frontend/src/`.

```bash
cd platform/frontend
npm test
```

## Adding New RAG Implementations

For a comprehensive guide on developing and integrating custom RAG systems, see the **[Custom RAG Integration Guide](docs/custom_rag_integration.md)**.

### Quick Overview

To add a new RAG strategy (e.g., "Elasticsearch RAG"):

1.  **Implement Core Logic:**
    Create `src/rag_evaluator/rag_implementations/elasticsearch_rag/`. Inherit from `BaseRAG` and implement the required methods: `prepare_documents()`, `query()`, and `get_metrics()`.

2.  **Register in CLI:**
    Update `src/rag_evaluator/cli.py` - add your RAG to `get_rag_implementation()` and the CLI argument choices.

3.  **Register in Backend (Optional):**
    If you want it available in the Platform, update `platform/backend/app/services/rag_adapter.py`:
    - Add to `RAG_TYPE_REGISTRY`
    - Add to `RAG_TYPE_PARAMETERS`
    - Add to `get_available_rag_types()`

### Developing as a Separate Project

For experimental or novel RAG systems, we recommend developing as a separate project first:

1. Create a standalone project with minimal dependencies
2. Copy the interface files (`base_rag.py`, `provider_interfaces.py`, `token_tracker.py`)
3. Develop and test your RAG independently
4. Integrate when stable by copying to `src/rag_evaluator/rag_implementations/`

See the [Custom RAG Integration Guide](docs/custom_rag_integration.md#developing-as-a-separate-project) for detailed instructions and a project template.

## Pull Request Process

1.  Create a feature branch.
2.  Ensure all tests pass and code is formatted.
3.  Submit PR with a clear description of changes.
4.  Wait for review.

---
**Happy Coding!**
