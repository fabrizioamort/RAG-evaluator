# RAG Evaluator

[![Tests](https://github.com/fabrizioamort/RAG-evaluator/workflows/Tests/badge.svg)](https://github.com/fabrizioamort/RAG-evaluator/actions)
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

## Supported Document Formats

The RAG Evaluator supports the following document formats:

- **TXT** - Plain text files
- **PDF** - PDF documents (pypdf)
- **DOCX** - Microsoft Word documents (python-docx)

Simply place your documents in `data/raw/` and run:

```bash
uv run rag-eval prepare --input-dir data/raw
```

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
uv run rag-eval prepare --input-dir data/raw

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

## Evaluation Framework

The project includes a comprehensive evaluation pipeline powered by [DeepEval](https://github.com/confident-ai/deepeval).

### Running Evaluations

```bash
# Prepare documents (one-time setup)
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw

# Run evaluation with default test set
uv run rag-eval evaluate --rag-type vector_semantic

# Run evaluation with custom test set
uv run rag-eval evaluate --rag-type vector_semantic --test-set my_tests.json

# Run with verbose output
uv run rag-eval evaluate --rag-type vector_semantic --verbose

# Alternative: use the evaluation script directly
uv run python scripts/run_evaluation.py --rag-type vector_semantic --verbose
```

### Test Dataset

The evaluation uses a test dataset (`data/test_set.json`) with question-answer pairs. Each test case includes:

- **question**: The query to test
- **expected_answer**: The ground truth answer
- **ground_truth_context**: Reference context chunks
- **difficulty**: Test case difficulty (easy/medium/hard)
- **category**: Question type (definition/explanation/comparison/etc.)

Example test case:

```json
{
  "id": "tc_001",
  "question": "What is RAG?",
  "expected_answer": "RAG (Retrieval Augmented Generation) combines...",
  "ground_truth_context": ["RAG is a technique that..."],
  "difficulty": "easy",
  "category": "definition"
}
```

### Evaluation Metrics

The framework evaluates RAG implementations across four key metrics:

1. **Faithfulness** (threshold: 0.7)
   - Measures if the answer is derived only from the retrieved context
   - Prevents hallucination

2. **Answer Relevancy** (threshold: 0.7)
   - Measures if the answer actually addresses the question
   - Ensures responses are on-topic

3. **Contextual Precision** (threshold: 0.7)
   - Measures if the retrieved documents are relevant
   - Evaluates retrieval quality

4. **Contextual Recall** (threshold: 0.7)
   - Measures if all relevant information was retrieved
   - Ensures comprehensive context

### Evaluation Reports

Each evaluation generates two report formats:

**JSON Report** (`reports/eval_<impl>_<timestamp>.json`)

- Machine-readable results
- Complete metric details
- Detailed per-test-case results

**Markdown Report** (`reports/eval_<impl>_<timestamp>.md`)

- Human-readable summary
- Metrics tables with pass/fail indicators
- Performance statistics
- Individual test case breakdowns

### Customizing Thresholds

Metric thresholds can be customized in `.env`:

```bash
EVAL_FAITHFULNESS_THRESHOLD=0.8
EVAL_ANSWER_RELEVANCY_THRESHOLD=0.75
EVAL_CONTEXTUAL_PRECISION_THRESHOLD=0.7
EVAL_CONTEXTUAL_RECALL_THRESHOLD=0.7
```

### Comparing Implementations

Compare multiple RAG implementations (coming soon):

```bash
uv run rag-eval evaluate --rag-type all
```

This generates a comparison report highlighting strengths and weaknesses of each approach.

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

**LLM Configuration:**

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_MODEL` - Model for answer generation (default: gpt-4-turbo-preview, supports gpt-5-nano)
- `EMBEDDING_MODEL` - Model for embeddings (default: text-embedding-3-small)
- `OPENAI_TIMEOUT` - API timeout in seconds (default: 600)

**Database Configuration:**

- `CHROMA_PERSIST_DIRECTORY` - ChromaDB storage location (default: ./data/chroma_db)
- `NEO4J_URI` - Neo4j connection URI (for Graph RAG)
- `NEO4J_PASSWORD` - Neo4j password

**Evaluation Configuration:**

- `EVAL_TEST_SET_PATH` - Path to test dataset (default: data/test_set.json)
- `EVAL_REPORTS_DIR` - Reports output directory (default: reports)
- `EVAL_FAITHFULNESS_THRESHOLD` - Faithfulness metric threshold (default: 0.7)
- `EVAL_ANSWER_RELEVANCY_THRESHOLD` - Answer relevancy threshold (default: 0.7)
- `EVAL_CONTEXTUAL_PRECISION_THRESHOLD` - Context precision threshold (default: 0.7)
- `EVAL_CONTEXTUAL_RECALL_THRESHOLD` - Context recall threshold (default: 0.7)

**DeepEval Configuration:**

- `DEEPEVAL_ASYNC_MODE` - Enable parallel evaluation (default: False, set to False to avoid rate limits)
- `DEEPEVAL_PER_TASK_TIMEOUT` - Per-task timeout in seconds (default: 900)
- `DEEPEVAL_PER_ATTEMPT_TIMEOUT` - Per-attempt timeout in seconds (default: 300)
- `DEEPEVAL_MAX_RETRIES` - Maximum retry attempts (default: 3)

## Platform Support

**Windows Compatibility:**

- CLI output optimized for Windows console (no emoji characters that cause encoding errors)
- Proper handling of file paths and line endings
- All features tested on Windows 10/11

**Model Compatibility:**

- Supports both standard OpenAI models (gpt-4-turbo, gpt-4o) and newer models (gpt-5-nano)
- Automatic temperature parameter adjustment for models that don't support it (e.g., gpt-5-nano)

**Rate Limiting:**

- Configurable async mode to prevent API rate limit errors
- Automatic retry logic with exponential backoff
- Timeout configuration for long-running evaluations

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
