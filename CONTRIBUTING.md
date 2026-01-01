# Contributing to RAG Evaluator

Thank you for your interest in contributing to RAG Evaluator! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [How to Contribute](#how-to-contribute)
- [Adding New RAG Implementations](#adding-new-rag-implementations)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## Code of Conduct

This project follows a simple principle: **Be respectful, be collaborative, and focus on improving the project.**

We welcome contributions from developers of all skill levels. If you're new to the project, look for issues labeled `good-first-issue`.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/rag-evaluator.git
   cd rag-evaluator
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/rag-evaluator.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher (3.12 recommended)
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Installation

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Verify Installation

```bash
# Run tests
uv run pytest

# Check code quality
uv run ruff check .

# Verify type hints
uv run mypy src/rag_evaluator
```

## Code Quality Standards

We maintain high code quality standards. All contributions must meet these requirements:

### Code Style

- **Formatter**: [Ruff](https://github.com/astral-sh/ruff) with 100 character line length
- **Linter**: Ruff with selected rules (E, F, I, N, W, UP)
- **Type Hints**: All functions must have complete type hints
- **Docstrings**: All public functions, classes, and modules must have docstrings

### Run Quality Checks

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Type checking
uv run mypy src/rag_evaluator

# Run all checks at once (recommended before committing)
make check
```

### Type Hints Requirements

All functions must include type hints:

```python
# ✅ Good
def process_document(path: str, chunk_size: int = 512) -> list[str]:
    """Process a document and return chunks."""
    pass

# ❌ Bad - missing type hints
def process_document(path, chunk_size=512):
    pass
```

For functions with no return value, use `-> None`:

```python
def save_results(data: dict[str, Any]) -> None:
    """Save results to disk."""
    pass
```

## Testing Requirements

All contributions must include tests. We use `pytest` for testing.

### Writing Tests

- **Unit tests** go in `tests/unit/`
- **Integration tests** go in `tests/integration/`
- Test files must start with `test_`
- Test functions must start with `test_`

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include a regression test

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/rag_evaluator --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_base_rag.py

# Run specific test function
uv run pytest tests/unit/test_base_rag.py::test_base_rag_initialization

# Using make
make test
```

## How to Contribute

### Reporting Bugs

Create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and environment details

### Suggesting Features

Create an issue with:
- Clear description of the feature
- Use case and benefits
- Potential implementation approach (optional)

### Contributing Code

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**:
   - Follow code quality standards
   - Include tests
   - Update documentation

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

   Use descriptive commit messages:
   - `Add feature: implement hybrid search RAG`
   - `Fix: correct ChromaDB connection handling`
   - `Docs: update README with new examples`

4. **Keep your fork updated**:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Adding New RAG Implementations

This is a common contribution type. Here's the complete process:

### Step 1: Create the Implementation Module

Create a new directory under `src/rag_evaluator/rag_implementations/`:

```bash
mkdir -p src/rag_evaluator/rag_implementations/your_rag_name
touch src/rag_evaluator/rag_implementations/your_rag_name/__init__.py
touch src/rag_evaluator/rag_implementations/your_rag_name/your_rag.py
```

### Step 2: Implement the BaseRAG Interface

```python
from typing import Any
from rag_evaluator.common.base_rag import BaseRAG


class YourRAG(BaseRAG):
    """Your RAG implementation description."""

    def __init__(self) -> None:
        """Initialize your RAG implementation."""
        super().__init__("Your RAG Name")
        # Initialize your components here

    def prepare_documents(self, documents_path: str) -> None:
        """Prepare and index documents.

        Args:
            documents_path: Path to the directory containing documents
        """
        # Implementation here
        pass

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query the RAG system.

        Args:
            question: The question to answer
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary containing:
                - answer: The generated answer
                - context: Retrieved context documents
                - metadata: Additional metadata (retrieval time, etc.)
        """
        # Implementation here
        return {
            "answer": "Your answer",
            "context": ["context1", "context2"],
            "metadata": {"retrieval_time": 0.0},
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary containing performance metrics
        """
        return {
            "avg_retrieval_time": 0.0,
            "total_documents": 0,
            # Add your specific metrics
        }
```

### Step 3: Add Tests

Create test files:

```bash
touch tests/unit/test_your_rag.py
touch tests/integration/test_your_rag_integration.py
```

Example unit test:

```python
from rag_evaluator.rag_implementations.your_rag_name.your_rag import YourRAG


def test_your_rag_initialization() -> None:
    """Test that YourRAG initializes correctly."""
    rag = YourRAG()
    assert rag.name == "Your RAG Name"


def test_your_rag_query_structure() -> None:
    """Test that query returns correct structure."""
    rag = YourRAG()
    result = rag.query("test question")

    assert "answer" in result
    assert "context" in result
    assert "metadata" in result
```

### Step 4: Update CLI

Add your implementation to `src/rag_evaluator/cli.py`:

```python
# In the evaluate command's rag-type choices
eval_parser.add_argument(
    "--rag-type",
    choices=[
        "vector_semantic",
        "vector_hybrid",
        "graph_rag",
        "filesystem_rag",
        "your_rag_name",  # Add this
        "all"
    ],
    default="all",
    help="RAG implementation to evaluate",
)
```

### Step 5: Update UI

Add to `src/rag_evaluator/ui/streamlit_app.py`:

```python
_rag_type = st.selectbox(
    "Select RAG Implementation",
    [
        "ChromaDB Semantic Search",
        "Hybrid Search",
        "Neo4j Graph RAG",
        "Filesystem RAG",
        "Your RAG Name",  # Add this
    ],
)
```

### Step 6: Add Dependencies (if needed)

If your implementation requires new packages:

```bash
# Add production dependency
uv add package-name

# Add dev dependency
uv add --dev package-name
```

### Step 7: Update Documentation

- Add description to README.md
- Update CLAUDE.md if needed
- Add example usage to docs

### Step 8: Run All Checks

```bash
# Format code
uv run ruff format .

# Run linter
uv run ruff check .

# Run type checker
uv run mypy src/rag_evaluator

# Run tests
uv run pytest

# Or use make
make check
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines (ruff format passes)
- [ ] All linting checks pass (ruff check passes)
- [ ] Type hints are complete (mypy passes)
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Commit messages are descriptive

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you added

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No breaking changes (or documented if necessary)
```

### Review Process

1. **Automated checks** will run on your PR
2. **Maintainer review** - expect feedback within a few days
3. **Address feedback** - make requested changes
4. **Approval and merge** - once approved, your PR will be merged

### After Your PR is Merged

- Delete your feature branch (optional)
- Update your local repository:
  ```bash
  git checkout master
  git pull upstream master
  ```

## Project Structure

Understanding the project structure will help you contribute effectively:

```
RAG-evaluator/
├── src/rag_evaluator/
│   ├── rag_implementations/   # RAG implementation modules
│   │   ├── vector_semantic/   # ChromaDB implementation
│   │   ├── vector_hybrid/     # Hybrid search implementation
│   │   ├── graph_rag/         # Neo4j Graph RAG
│   │   └── filesystem_rag/    # Filesystem-based RAG
│   ├── evaluation/            # Evaluation framework using DeepEval
│   ├── common/                # Shared utilities (BaseRAG, etc.)
│   ├── ui/                    # Streamlit web interface
│   ├── config.py              # Pydantic settings management
│   └── cli.py                 # CLI entry point
├── tests/
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── data/
│   ├── raw/                   # Source documents (gitignored)
│   └── processed/             # Processed documents (gitignored)
├── reports/                   # Evaluation reports (gitignored)
└── scripts/                   # Helper scripts
```

### Key Design Patterns

- **Abstract Base Class**: All RAG implementations inherit from `BaseRAG`
- **Configuration**: Uses Pydantic Settings with `.env` file
- **Evaluation**: Standardized evaluation through `RAGEvaluator` class
- **Type Safety**: Comprehensive type hints throughout

## Questions?

- Check existing issues and discussions
- Review [CLAUDE.md](CLAUDE.md) for detailed development guidance
- Create a new issue with the `question` label

## Thank You!

Your contributions make this project better. We appreciate your time and effort! 🙏
