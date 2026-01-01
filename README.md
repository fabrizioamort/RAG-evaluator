# RAG Evaluator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://github.com/python/mypy)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://github.com/pytest-dev/pytest)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A comprehensive evaluation framework for comparing different RAG (Retrieval Augmented Generation) methodologies and technologies.

## Overview

This project implements and evaluates four different RAG approaches:

1. **Vector Semantic Search** - Using ChromaDB for pure semantic similarity
2. **Hybrid Search** - Combining semantic and keyword-based retrieval
3. **Graph RAG** - Using Neo4j graph database with LangChain
4. **Filesystem RAG** - Direct filesystem search with LLM-guided retrieval

## Evaluation Metrics

Each RAG implementation is evaluated on:

- **Accuracy**
  - Faithfulness: Is the answer derived only from the context?
  - Answer Relevance: Does it actually answer the user's question?
  - Context Precision: Did the retriever find the right documents?
- **Speed** - Query and retrieval performance
- **Cost** - API calls and resource usage

Evaluation is powered by [DeepEval](https://github.com/confident-ai/deepeval).

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repository-url>
cd RAG-evaluator

# Install dependencies
uv sync --all-extras

# Copy environment template
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Quick Start

### Using the CLI

```bash
# Prepare documents for RAG implementations
uv run rag-eval prepare --input-dir data/raw --output-dir data/processed

# Run evaluation on a specific RAG implementation
uv run rag-eval evaluate --rag-type vector_semantic

# Run evaluation on all implementations
uv run rag-eval evaluate --rag-type all --output reports

# Launch the web UI
uv run rag-eval ui
```

### Using the Streamlit UI

```bash
# Alternative way to launch the UI
uv run python scripts/run_streamlit.py
```

## Project Structure

```
RAG-evaluator/
├── src/rag_evaluator/
│   ├── rag_implementations/     # RAG implementation modules
│   │   ├── vector_semantic/     # ChromaDB semantic search
│   │   ├── vector_hybrid/       # Hybrid search
│   │   ├── graph_rag/           # Neo4j graph RAG
│   │   └── filesystem_rag/      # Filesystem-based RAG
│   ├── evaluation/              # Evaluation framework
│   ├── common/                  # Shared utilities and base classes
│   ├── ui/                      # Streamlit web interface
│   ├── config.py               # Configuration management
│   └── cli.py                  # CLI entry point
├── data/
│   ├── raw/                    # Source documents
│   └── processed/              # Processed documents
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── reports/                    # Evaluation reports
└── scripts/                    # Helper scripts
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/rag_evaluator --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_evaluator.py
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/rag_evaluator
```

## Configuration

Key configuration options in `.env`:

- `OPENAI_API_KEY` - Your OpenAI API key
- `OPENAI_MODEL` - Model for answer generation (default: gpt-4-turbo-preview)
- `EMBEDDING_MODEL` - Model for embeddings (default: text-embedding-3-small)
- `NEO4J_URI` - Neo4j connection URI
- `NEO4J_PASSWORD` - Neo4j password

## Requirements

- Python 3.11+
- OpenAI API key
- Neo4j database (for Graph RAG implementation)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code quality standards and testing requirements
- How to add new RAG implementations
- Pull request process

## License

MIT
