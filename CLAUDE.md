# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Evaluator is a Python project that compares and evaluates different RAG (Retrieval Augmented Generation) implementations. It uses **uv** for dependency management and provides both CLI and Streamlit UI interfaces.

The project evaluates four RAG approaches:

1. Vector semantic search (ChromaDB)
2. Hybrid search (semantic + keyword)
3. Graph RAG (Neo4j + LangChain)
4. Filesystem RAG (LLM-guided file retrieval)

## Development Commands

### Environment Setup

```bash
# Install all dependencies (including dev tools)
uv sync --all-extras

# Install only production dependencies
uv sync

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name
```

### Running the Application

```bash
# Run CLI commands
uv run rag-eval prepare --input-dir data/raw
uv run rag-eval evaluate --rag-type all
uv run rag-eval ui

# Run Streamlit UI directly
uv run python scripts/run_streamlit.py
```

### Testing

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/rag_evaluator --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/test_evaluator.py

# Run specific test function
uv run pytest tests/unit/test_evaluator.py::test_function_name

# Run integration tests only
uv run pytest tests/integration/
```

### Code Quality

```bash
# Format code with ruff
uv run ruff format .

# Lint code
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Type checking
uv run mypy src/rag_evaluator

# Run all quality checks (recommended before commits)
uv run ruff format . && uv run ruff check . && uv run mypy src/rag_evaluator
```

## Architecture

### Core Design Pattern

The project uses an **abstract base class pattern** for RAG implementations:

- `BaseRAG` (src/rag_evaluator/common/base_rag.py) defines the interface
- Each RAG implementation inherits from `BaseRAG` and implements:
  - `prepare_documents()` - Index/prepare documents
  - `query()` - Execute retrieval and generation
  - `get_metrics()` - Return performance metrics

### Key Components

**RAG Implementations** (`src/rag_evaluator/rag_implementations/`):

- Each subdirectory contains a specific RAG approach
- All implementations follow the `BaseRAG` interface
- Implementations should be self-contained within their module

**Evaluation Framework** (`src/rag_evaluator/evaluation/`):

- `evaluator.py` - Main evaluation logic using DeepEval with 4 metrics:
  - Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall
- `report_generator.py` - Generates JSON and Markdown evaluation reports
- Loads test cases from JSON dataset (default: `data/test_set.json`)
- Supports single implementation evaluation or multi-implementation comparison
- All metric thresholds configurable via `.env`
- Reports saved to `reports/` directory with timestamps

**Configuration** (`src/rag_evaluator/config.py`):

- Uses Pydantic Settings for environment-based configuration
- All settings loaded from `.env` file
- Type-safe configuration access via `settings` object

**CLI** (`src/rag_evaluator/cli.py`):

- Entry point defined in pyproject.toml as `rag-eval`
- Three main commands: `prepare`, `evaluate`, `ui`
- Uses argparse for command-line parsing

**UI** (`src/rag_evaluator/ui/streamlit_app.py`):

- Streamlit-based web interface
- Three tabs: Query, Evaluate, Compare
- Can be launched via CLI or directly

### Data Flow

1. **Document Preparation**: Raw documents → Processing → Indexed/prepared format
2. **Query Execution**: Question → RAG retrieval → LLM generation → Answer + context
3. **Evaluation**: Test cases → Multiple queries → DeepEval metrics → Reports

## Adding New RAG Implementations

To add a new RAG implementation:

1. Create new directory under `src/rag_evaluator/rag_implementations/`
2. Create implementation class inheriting from `BaseRAG`
3. Implement required methods: `prepare_documents()`, `query()`, `get_metrics()`
4. Add to CLI choices in `cli.py`
5. Add to UI options in `ui/streamlit_app.py`
6. Create unit tests in `tests/unit/`
7. Create integration tests in `tests/integration/`

Example structure:

```python
from rag_evaluator.common.base_rag import BaseRAG

class NewRAG(BaseRAG):
    def __init__(self) -> None:
        super().__init__("New RAG Implementation")
        # Initialize your components

    def prepare_documents(self, documents_path: str) -> None:
        # Implement document preparation
        pass

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        # Implement query logic
        return {
            "answer": "...",
            "context": [...],
            "metadata": {"retrieval_time": 0.0}
        }

    def get_metrics(self) -> dict[str, Any]:
        # Return performance metrics
        return {...}
```

## Type Hints and Code Quality

- **All functions require type hints** - Use `-> None` for functions without return values
- **mypy strict mode** - `disallow_untyped_defs = true` in pyproject.toml
- **Line length** - Maximum 100 characters (configured in ruff)
- **Import sorting** - Automatically handled by ruff (using "I" rule)

## Testing Conventions

- Unit tests in `tests/unit/` - Test individual components in isolation
- Integration tests in `tests/integration/` - Test component interactions
- Test files must start with `test_`
- Test functions must start with `test_`
- Use pytest fixtures for common setup
- Aim for high code coverage (configured to show missing lines)

## Configuration Management

- Never commit `.env` file (use `.env.example` as template)
- Access configuration via `from rag_evaluator.config import settings`
- Add new settings to `Settings` class in `config.py`
- Update `.env.example` when adding new configuration options

## Common Development Workflows

### Implementing a RAG Feature

1. Read the base class to understand the interface
2. Check existing implementations for patterns
3. Implement the new feature following the established pattern
4. Add appropriate error handling
5. Update type hints and docstrings
6. Write tests for the new functionality
7. Run quality checks before committing

### Running Evaluations

**Basic Evaluation:**

```bash
# Prepare documents first (one-time)
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw

# Run evaluation
uv run rag-eval evaluate --rag-type vector_semantic

# Run with verbose output
uv run rag-eval evaluate --rag-type vector_semantic --verbose
```

**Custom Test Sets:**

```bash
# Use custom test set
uv run rag-eval evaluate --test-set path/to/my_tests.json

# Save reports to custom directory
uv run rag-eval evaluate --output my_reports/
```

**Using the Script:**

```bash
# Alternative approach with more options
uv run python scripts/run_evaluation.py --rag-type vector_semantic --verbose
```

### Debugging Evaluation Issues

1. **Test Set Format**: Verify JSON structure matches expected format:
   - Must have `test_cases` array
   - Each case needs: `question`, `expected_answer`, `ground_truth_context`

2. **RAG Response Structure**: Ensure `query()` returns:

   ```python
   {
       "answer": str,
       "context": list[str],  # List of context chunks
       "metadata": {"retrieval_time": float}
   }
   ```

3. **DeepEval Metrics**: Each metric requires:
   - `input`: The question
   - `actual_output`: RAG's answer
   - `expected_output`: Ground truth answer
   - `retrieval_context`: List of retrieved chunks

4. **API Issues**:
   - Check `OPENAI_API_KEY` is set in `.env`
   - Verify API rate limits and quotas
   - Use `--verbose` flag to see detailed API calls

5. **Metric Failures**:
   - Review threshold settings in `.env`
   - Check individual test case results in detailed_results
   - Lower thresholds if too strict for your use case

### Working with Dependencies

- Core dependencies (langchain, chromadb, etc.) in main `dependencies`
- Development tools (pytest, ruff, mypy) in `[project.optional-dependencies]`
- Pin major versions, allow minor/patch updates (e.g., `>=0.1.0`)
- Test after adding new dependencies to ensure compatibility

## File Organization Principles

- **src/rag_evaluator/** - All application code
- **tests/** - Mirror the src structure in tests
- **data/raw/** - Source documents (gitignored)
- **data/processed/** - Prepared data (gitignored)
- **reports/** - Evaluation outputs (gitignored)
- **scripts/** - Helper scripts and tools

## Important Notes

- This is a showcase/portfolio project - code quality and documentation matter
- Each RAG implementation should be independently runnable
- Evaluation results should be reproducible
- The Streamlit UI should provide clear visualization of comparisons
- Keep dependencies minimal and well-justified
- Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.
