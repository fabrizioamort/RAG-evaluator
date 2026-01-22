<p align="center">
  <img src="docs/images/logo.png" alt="RAG Evaluator Logo" width="120"/>
</p>
<!-- PLACEHOLDER: logo.png - A professional logo for the RAG Evaluator Platform (recommended: 120x120px PNG with transparent background) -->

<h1 align="center">RAG Evaluator Platform</h1>

<p align="center">
  <strong>The comprehensive platform for designing, testing, and evaluating Retrieval Augmented Generation systems.</strong>
</p>

<p align="center">
  <a href="https://github.com/fabrizioamort/RAG-evaluator/actions"><img src="https://github.com/fabrizioamort/RAG-evaluator/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code style: ruff"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/docker-ready-blue.svg" alt="Docker"></a>
</p>

<p align="center">
  <a href="#-features">Features</a> &bull;
  <a href="#-quick-start">Quick Start</a> &bull;
  <a href="#-documentation">Documentation</a> &bull;
  <a href="#-architecture">Architecture</a> &bull;
  <a href="#-contributing">Contributing</a>
</p>

---

<p align="center">
  <img src="docs/images/hero-screenshot.png" alt="RAG Evaluator Dashboard" width="800"/>
</p>
<!-- PLACEHOLDER: hero-screenshot.png - Main dashboard screenshot showing the platform UI (recommended: 1600x900px) -->

---

## Overview

RAG Evaluator Platform provides a unified environment to experiment with different retrieval strategies, manage knowledge bases, and rigorously benchmark performance using the [DeepEval](https://github.com/confident-ai/deepeval) framework.

Whether you're building a chatbot, a search engine, or an AI assistant, this platform helps you:

- **Compare** different RAG architectures side-by-side
- **Measure** quality with industry-standard metrics
- **Debug** issues with detailed explainability
- **Track** improvements over time

---

## Features

### Enterprise-Grade Platform

<table>
<tr>
<td width="50%">

**Project Management**

Organize evaluations by project and use case. Track multiple experiments with version-controlled test sets and configurations.

</td>
<td width="50%">

**Knowledge Base Management**

Upload and index documents (PDF, DOCX, TXT) with ease. Build multiple indexes with different RAG strategies for comparison.

</td>
</tr>
<tr>
<td width="50%">

**Visual Analytics**

Interactive dashboards to track performance trends over time. Identify regressions early and celebrate improvements.

</td>
<td width="50%">

**Metric Explainability**

Understand *why* a score is low with detailed reasoning from the LLM judge. Get actionable recommendations for improvement.

</td>
</tr>
</table>

### Advanced RAG Architectures

Compare multiple implementation strategies out-of-the-box:

| Strategy | Database | Best For |
|----------|----------|----------|
| **Vector Semantic** | ChromaDB | General Q&A, semantic matching |
| **Hybrid Search** | Qdrant | Technical docs, keyword + semantic |
| **Graph RAG** | Neo4j | Relationship queries, multi-hop reasoning |
| **Filesystem RAG** | Agentic | Large doc sets, "needle in haystack" |

> See the [RAG Strategies Guide](docs/rag_strategies.md) for detailed architecture and configuration.

### Rigorous Evaluation

Powered by [DeepEval](https://github.com/confident-ai/deepeval), evaluations use LLM-as-judge methodology:

| Metric | What It Measures | Key Question |
|--------|------------------|--------------|
| **Faithfulness** | Hallucinations | Is the answer grounded in context? |
| **Answer Relevancy** | Utility | Does it address the question? |
| **Contextual Precision** | Ranking | Are relevant chunks ranked first? |
| **Contextual Recall** | Completeness | Did we find all needed info? |
| **Correctness (G-Eval)** | Accuracy | Is it semantically correct? |

> See the [Metrics Guide](docs/metrics.md) for detailed definitions and usage strategies.

---

## Quick Start

### Option A: Web Platform (Recommended)

The full-stack application for teams and production use.

```bash
# 1. Clone the repository
git clone https://github.com/fabrizioamort/RAG-evaluator.git
cd RAG-evaluator

# 2. Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 3. Launch with Docker
docker-compose up -d
```

**Access:**
- **Dashboard:** [http://localhost:3000](http://localhost:3000)
- **API Docs:** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

### Option B: CLI Tool

For developers and CI/CD integration.

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env

# 3. Prepare and evaluate
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw
uv run rag-eval evaluate --rag-type vector_semantic
```

> See the [Getting Started Guide](docs/guides/getting-started.md) for detailed tutorials.

---

## Architecture

The platform consists of three tiers sharing a common core engine:

```mermaid
graph TD
    User[User] -->|Web Browser| Frontend[React Frontend]
    User -->|Terminal| CLI[CLI Tool]

    Frontend -->|REST API| Backend[FastAPI Backend]

    subgraph Core["Core Engine"]
        Backend --> RAG[RAG Implementations]
        CLI --> RAG
        RAG --> Eval[DeepEval Framework]
    end

    subgraph Storage["Storage Layer"]
        RAG --> Chroma[(ChromaDB)]
        RAG --> Qdrant[(Qdrant)]
        RAG --> Neo4j[(Neo4j)]
        Backend --> DB[(PostgreSQL)]
    end
```

> See the [Architecture Documentation](docs/ARCHITECTURE.md) for detailed component diagrams and data flows.

---

## Documentation

### Guides

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/guides/getting-started.md) | Your first evaluation in 10 minutes |
| [Evaluation Guide](docs/guides/evaluation-guide.md) | Test design and result interpretation |
| [UI Guide](docs/guides/ui-guide.md) | Complete walkthrough of the web interface |
| [Configuration](docs/guides/configuration.md) | All environment variables explained |
| [Troubleshooting](docs/guides/troubleshooting.md) | Common issues and solutions |
| [Security](docs/guides/security.md) | Production deployment best practices |

### Reference

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design and component relationships |
| [RAG Strategies](docs/rag_strategies.md) | Detailed guide to each RAG implementation |
| [Metrics](docs/metrics.md) | Evaluation metrics deep dive |
| [API Reference](docs/api.md) | Complete REST API documentation |
| [CLI Reference](docs/cli.md) | Command-line usage and options |
| [Custom RAG](docs/custom_rag_integration.md) | Build and integrate your own RAG |
| [Deployment](docs/deployment.md) | Production deployment instructions |

---

## Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Docker & Docker Compose

### Setup

```bash
# Install all dependencies
make install

# Or install individually
uv sync --all-extras              # Core library
cd platform/backend && uv sync    # Backend
cd platform/frontend && npm install  # Frontend
```

### Running Locally

```bash
# Start databases
make dev-infra

# Start backend (separate terminal)
make dev-backend

# Start frontend (separate terminal)
make dev-frontend
```

### Testing

```bash
# Run all tests
make test

# Run specific suites
make test-core      # Core library (pytest)
make test-backend   # Backend API (pytest)
make test-frontend  # Frontend (vitest)
```

### Code Quality

```bash
# Run all linters
make lint

# Format code
make format

# Pre-push check
make check          # format → lint → test
```

> See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

---

## Configuration

Key configuration via `.env`:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key

# LLM Settings
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Database (optional - defaults to SQLite)
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db

# Vector Stores
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
```

> See [Configuration Reference](docs/guides/configuration.md) for all options.

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Code quality standards
- Testing requirements
- Adding new RAG implementations
- Pull request process

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with care for the RAG community
</p>

<p align="center">
  <a href="https://github.com/fabrizioamort/RAG-evaluator/stargazers">Star us on GitHub</a>
</p>
