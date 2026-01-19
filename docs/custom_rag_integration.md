# Custom RAG Integration Guide

This guide explains how to develop your own RAG (Retrieval-Augmented Generation) system and integrate it with the RAG Evaluator platform. Whether you're experimenting with a novel retrieval strategy or adapting an existing system for evaluation, this document provides a complete roadmap.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Development Approaches](#development-approaches)
- [The BaseRAG Interface](#the-baserag-interface)
- [Step-by-Step Integration](#step-by-step-integration)
- [Developing as a Separate Project](#developing-as-a-separate-project)
- [Configuration Parameters](#configuration-parameters)
- [Complete Integration Checklist](#complete-integration-checklist)
- [Testing Your Implementation](#testing-your-implementation)
- [Linting and Code Quality](#linting-and-code-quality)
- [Platform UI Integration](#platform-ui-integration)
- [Best Practices](#best-practices)

---

## Overview

The RAG Evaluator uses a clean abstraction layer that allows any RAG implementation to be plugged in and evaluated using the same metrics (Faithfulness, Answer Relevancy, Contextual Precision, etc.). All RAG systems inherit from a common `BaseRAG` class and implement a standard interface.

**Key Benefits:**

- **Unified Evaluation:** Compare your custom RAG against built-in implementations using identical test sets and metrics.
- **Modular Design:** Each RAG implementation is self-contained in its own directory.
- **Low Integration Overhead:** Adding a new RAG type requires only 3-4 lines of registration code.

---

## Architecture

### How RAG Systems Fit In

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Evaluator Platform                      │
├─────────────────────────────────────────────────────────────────┤
│  CLI Tool                           │  Web Platform (FastAPI)   │
│  ├─ prepare command                 │  ├─ /api/evaluations      │
│  ├─ evaluate command                │  ├─ /api/knowledge-bases  │
│  └─ ui command                      │  └─ /api/rag-configs      │
├─────────────────────────────────────────────────────────────────┤
│                         Core Engine                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    BaseRAG Interface                         ││
│  │  ┌────────────┬────────────┬────────────┬────────────────┐  ││
│  │  │  Vector    │  Hybrid    │  Graph     │  Your Custom   │  ││
│  │  │  Semantic  │  Search    │  RAG       │  RAG           │  ││
│  │  └────────────┴────────────┴────────────┴────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│                         DeepEval Framework                       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `src/rag_evaluator/common/base_rag.py` | Abstract base class all RAGs inherit from |
| `src/rag_evaluator/common/provider_interfaces.py` | Data classes for retrieval/generation results |
| `src/rag_evaluator/cli.py` | CLI registration (factory function) |
| `platform/backend/app/services/rag_adapter.py` | Platform registration (dynamic import) |

---

## Development Approaches

You have two main options for developing a custom RAG:

### Option A: Develop Inside the Codebase

**Best for:** Quick experiments, minor variations of existing implementations.

```
src/rag_evaluator/rag_implementations/
├── vector_semantic/
├── vector_hybrid/
├── graph_rag/
├── filesystem_rag/
└── my_custom_rag/          # Add your implementation here
    ├── __init__.py
    └── my_rag.py
```

### Option B: Develop as a Separate Project (Recommended)

**Best for:** Novel or experimental RAG systems, cleaner iteration cycles.

Develop your RAG in isolation, then copy it into the codebase when ready. This approach:

- Avoids inheriting all project dependencies during prototyping
- Allows faster iteration without running the full platform
- Makes it easy to test independently before integration

See [Developing as a Separate Project](#developing-as-a-separate-project) for a detailed guide.

---

## The BaseRAG Interface

Every RAG implementation must inherit from `BaseRAG` and implement these abstract methods:

### Required Methods

```python
from rag_evaluator.common.base_rag import BaseRAG, RAGConfig
from typing import Any

class MyCustomRAG(BaseRAG):
    def __init__(self, config: RAGConfig | None = None, **kwargs):
        super().__init__(name="My Custom RAG", config=config)
        # Initialize your RAG-specific components

    def prepare_documents(self, documents_path: str) -> None:
        """Index documents for retrieval.

        Args:
            documents_path: Path to directory containing documents (PDF, DOCX, TXT)
        """
        # Your indexing logic here
        pass

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Execute full RAG pipeline (retrieve + generate).

        Args:
            question: The user's question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with keys:
                - answer: str - The generated answer
                - context: list[str] - Retrieved context chunks
                - metadata: dict - Retrieval time, token usage, etc.
        """
        # Your retrieval + generation logic here
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Return performance metrics.

        Returns:
            Dictionary with implementation-specific metrics
            (e.g., index size, retrieval times, chunk counts)
        """
        pass
```

### Optional Methods (Recommended)

For better observability and flexibility, override these methods:

```python
from rag_evaluator.common.provider_interfaces import (
    RetrievedContext,
    RetrievedChunk,
    RetrievalTrace,
    GeneratedAnswer,
)

class MyCustomRAG(BaseRAG):
    # ... required methods ...

    def retrieve(self, question: str, top_k: int = 5) -> RetrievedContext:
        """Retrieval only (no generation).

        Enables caching retrieval results and running generation experiments.
        """
        trace = RetrievalTrace(strategy="my_strategy")

        # Your retrieval logic
        chunks = [...]
        chunk_details = [
            RetrievedChunk(
                content=chunk,
                document_id="doc_1",
                chunk_id="chunk_1",
                score=0.95,
                rank=0,
                source="document.pdf",
            )
            for chunk in chunks
        ]

        trace.retrieved_chunks = chunk_details

        return RetrievedContext(
            chunks=[c.content for c in chunk_details],
            chunk_details=chunk_details,
            trace=trace,
            retrieval_time=0.5,  # seconds
        )

    def generate(self, question: str, context: RetrievedContext) -> GeneratedAnswer:
        """Generation only (from pre-retrieved context).

        Enables prompt experiments without re-retrieval.
        """
        # Your generation logic using context.chunks
        answer_text = "..."

        return GeneratedAnswer(
            text=answer_text,
            generation_time=1.2,
            prompt_tokens=500,
            completion_tokens=150,
        )

    def _get_strategy_name(self) -> str:
        """Return strategy name for tracing."""
        return "my_custom_strategy"  # e.g., "vector", "hybrid", "graph", "agentic"

    def close(self) -> None:
        """Clean up resources (database connections, etc.)."""
        pass
```

### Token Tracking

The base class provides thread-safe token tracking. Use it like this:

```python
def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
    self.reset_token_usage()  # Reset per-query counters

    # After LLM calls, track tokens:
    self._token_usage.add_prompt_tokens(prompt_tokens)
    self._token_usage.add_completion_tokens(completion_tokens)
    self._token_usage.add_embedding_tokens(embedding_tokens)

    return {
        "answer": answer,
        "context": context_chunks,
        "metadata": {
            "token_usage": self._token_usage.to_dict(),
            # ...
        }
    }
```

---

## Step-by-Step Integration

### Step 1: Create Your Implementation

Create a new directory under `src/rag_evaluator/rag_implementations/`:

```
src/rag_evaluator/rag_implementations/my_custom_rag/
├── __init__.py
└── my_rag.py
```

**`__init__.py`:**
```python
from .my_rag import MyCustomRAG

__all__ = ["MyCustomRAG"]
```

**`my_rag.py`:**
```python
"""My Custom RAG implementation."""

from typing import Any
from rag_evaluator.common.base_rag import BaseRAG, RAGConfig


class MyCustomRAG(BaseRAG):
    """A custom RAG implementation using [describe your approach]."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        # Add your custom parameters here
        my_param: str = "default_value",
    ) -> None:
        super().__init__(name="My Custom RAG", config=config)
        self.my_param = my_param
        # Initialize your components

    def prepare_documents(self, documents_path: str) -> None:
        """Index documents."""
        # Implementation
        pass

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query the RAG system."""
        # Implementation
        return {
            "answer": "...",
            "context": ["chunk1", "chunk2"],
            "metadata": {"retrieval_time": 0.5},
        }

    def get_metrics(self) -> dict[str, Any]:
        """Return metrics."""
        return {
            "index_size": 0,
            "total_chunks": 0,
        }
```

### Step 2: Register in CLI

Edit `src/rag_evaluator/cli.py`:

```python
# Add import at the top
from rag_evaluator.rag_implementations.my_custom_rag import MyCustomRAG

# Add to get_rag_implementation() function
def get_rag_implementation(rag_type: str) -> BaseRAG:
    if rag_type == "vector_semantic":
        return ChromaSemanticRAG()
    elif rag_type == "vector_hybrid":
        return HybridSearchRAG()
    elif rag_type == "graph_rag":
        return Neo4jGraphRAG()
    elif rag_type == "filesystem_rag":
        return FilesystemRAG()
    elif rag_type == "my_custom_rag":        # <-- Add this
        return MyCustomRAG()
    else:
        raise ValueError(f"RAG type '{rag_type}' not yet implemented...")

# Update the CLI argument choices
prepare_parser.add_argument(
    "--rag-type",
    choices=["vector_semantic", "vector_hybrid", "graph_rag", "filesystem_rag", "my_custom_rag"],
    # ...
)
```

### Step 3: Register in Platform (Optional)

If you want your RAG available in the web UI, edit `platform/backend/app/services/rag_adapter.py`:

```python
# Add to RAG_TYPE_REGISTRY
RAG_TYPE_REGISTRY: dict[str, str] = {
    "vector_semantic": "rag_evaluator.rag_implementations.vector_semantic.chroma_rag.ChromaSemanticRAG",
    "vector_hybrid": "rag_evaluator.rag_implementations.vector_hybrid.hybrid_rag.HybridSearchRAG",
    "graph_rag": "rag_evaluator.rag_implementations.graph_rag.neo4j_rag.Neo4jGraphRAG",
    "filesystem_rag": "rag_evaluator.rag_implementations.filesystem_rag.filesystem_rag.FilesystemRAG",
    "my_custom_rag": "rag_evaluator.rag_implementations.my_custom_rag.my_rag.MyCustomRAG",  # <-- Add
}

# Add to RAG_TYPE_PARAMETERS (defines UI form fields)
RAG_TYPE_PARAMETERS: dict[str, dict[str, Any]] = {
    # ... existing entries ...
    "my_custom_rag": {
        "properties": {
            "my_param": {
                "type": "string",
                "default": "default_value",
                "description": "Description of this parameter",
            },
        },
    },
}

# Add to get_available_rag_types() method
def get_available_rag_types(self) -> list[dict[str, Any]]:
    return [
        # ... existing entries ...
        {
            "type": "my_custom_rag",
            "name": "My Custom RAG",
            "description": "Description of your RAG approach",
        },
    ]
```

### Step 4: Add Dependencies

If your RAG requires new Python packages, add them to `pyproject.toml`:

```toml
[project.dependencies]
# ... existing dependencies ...
my-new-package = ">=1.0.0"
```

Then run: `uv sync`

---

## Configuration Parameters

When your RAG has configurable parameters (chunk size, model names, thresholds, etc.), you need to define them so the UI can generate form fields and pass values to your constructor.

### How Parameters Flow Through the System

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend UI                                  │
│  RAGConfigDialog.tsx                                                │
│  - Fetches parameter definitions from GET /api/rag-types           │
│  - Dynamically generates form fields based on parameter type        │
│  - Sends parameters dict on form submit                             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ POST { parameters: { my_param: "x", max_depth: 10 } }
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Backend API                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  rag_registry.py - Defines UI form schema                    │   │
│  │  rag_adapter.py - Maps parameters → constructor kwargs       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Database (RAGConfig model)                                  │   │
│  │  - Stores parameters as JSON: {"my_param": "x", ...}         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ MyCustomRAG(my_param="x", max_depth=10)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Your RAG Implementation                          │
│  class MyCustomRAG(BaseRAG):                                        │
│      def __init__(self, my_param, max_depth, ...):                  │
│          self.my_param = my_param                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 1: Define Parameters in Registry

Edit `platform/backend/app/services/rag_registry.py`:

```python
RAGTypeInfo(
    name="my_custom_rag",
    display_name="My Custom RAG",
    description="Description shown in the UI dropdown",
    requires_index=True,
    parameters=[
        RAGTypeParameter(
            name="chunk_size",
            type="integer",
            description="Size of text chunks for indexing",
            default=1000,
            min_value=100,
            max_value=4000,
        ),
        RAGTypeParameter(
            name="search_strategy",
            type="string",
            description="Search algorithm to use",
            choices=["breadth_first", "depth_first", "hybrid"],  # Creates dropdown
            default="hybrid",
        ),
        RAGTypeParameter(
            name="enable_caching",
            type="boolean",
            description="Cache retrieval results",
            default=True,
        ),
        RAGTypeParameter(
            name="similarity_threshold",
            type="float",
            description="Minimum similarity score",
            default=0.7,
            min_value=0.0,
            max_value=1.0,
        ),
        RAGTypeParameter(
            name="api_endpoint",
            type="string",
            description="Custom API endpoint URL",
            required=False,  # Optional parameter
        ),
    ],
)
```

### Step 2: Map Parameters in Adapter

Edit `platform/backend/app/services/rag_adapter.py` in `create_rag_from_config()`:

```python
elif config_model.rag_type == "my_custom_rag":
    kwargs["chunk_size"] = config_model.parameters.get("chunk_size", 1000)
    kwargs["search_strategy"] = config_model.parameters.get("search_strategy", "hybrid")
    kwargs["enable_caching"] = config_model.parameters.get("enable_caching", True)
    kwargs["similarity_threshold"] = config_model.parameters.get("similarity_threshold", 0.7)
    kwargs["api_endpoint"] = config_model.parameters.get("api_endpoint")
```

Also update `create_rag_for_index()` with the same parameter mapping.

### Step 3: Accept Parameters in Constructor

```python
class MyCustomRAG(BaseRAG):
    def __init__(
        self,
        config: RAGConfig | None = None,
        chunk_size: int = 1000,
        search_strategy: str = "hybrid",
        enable_caching: bool = True,
        similarity_threshold: float = 0.7,
        api_endpoint: str | None = None,
    ) -> None:
        super().__init__(name="My Custom RAG", config=config)
        self.chunk_size = chunk_size
        self.search_strategy = search_strategy
        self.enable_caching = enable_caching
        self.similarity_threshold = similarity_threshold
        self.api_endpoint = api_endpoint
```

### Supported Parameter Types

| Type | UI Control | Python Type | Example |
|------|-----------|-------------|---------|
| `string` | Text input | `str` | `"default_value"` |
| `string` + `choices` | Dropdown select | `str` | `["option1", "option2"]` |
| `integer` | Number input (step=1) | `int` | `10` |
| `float` | Number input (step=0.1) | `float` | `0.5` |
| `boolean` | Checkbox | `bool` | `True` |

### Parameter Options

| Option | Description |
|--------|-------------|
| `name` | Parameter identifier (used in code) |
| `type` | Data type: `string`, `integer`, `float`, `boolean` |
| `description` | Help text shown below the input |
| `default` | Pre-filled value in form |
| `required` | Shows "Required" badge (default: `False`) |
| `min_value` | Minimum allowed value (numeric types) |
| `max_value` | Maximum allowed value (numeric types) |
| `choices` | List of allowed values (creates dropdown) |

---

## Developing as a Separate Project

This is the recommended approach for novel or experimental RAG systems.

### Why Develop Separately?

1. **Cleaner Iteration:** No need to run the full platform during development
2. **Minimal Dependencies:** Start with only what you need
3. **Independent Testing:** Validate your RAG before integration
4. **Version Control:** Keep your experimental code separate until stable

### Project Structure Template

Create a new directory **outside** the RAG Evaluator codebase:

```
my-custom-rag/
├── pyproject.toml
├── README.md
├── src/
│   └── my_custom_rag/
│       ├── __init__.py
│       ├── rag.py              # Your RAG implementation
│       └── base_rag.py         # Copy of BaseRAG interface
├── tests/
│   └── test_rag.py
└── examples/
    └── basic_usage.py
```

### Minimal Dependencies

Your `pyproject.toml` only needs:

```toml
[project]
name = "my-custom-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.12.0",          # For LLM/embeddings (if using OpenAI)
    # Add only what YOUR RAG needs
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
]
```

### Copy the Interface Files

Copy these files from the RAG Evaluator to your project:

```
# From: src/rag_evaluator/common/
# To:   my-custom-rag/src/my_custom_rag/

- base_rag.py
- provider_interfaces.py
- token_tracker.py
```

Or install RAG Evaluator as a dependency:

```toml
dependencies = [
    "rag-evaluator",  # If published to PyPI
]
```

### Integration Checklist

When your RAG is ready for integration:

1. **Copy implementation:**
   ```
   my-custom-rag/src/my_custom_rag/rag.py
   → RAG-evaluator/src/rag_evaluator/rag_implementations/my_custom_rag/my_rag.py
   ```

2. **Update imports:** Change from local interface to project interface:
   ```python
   # Before (separate project)
   from my_custom_rag.base_rag import BaseRAG

   # After (integrated)
   from rag_evaluator.common.base_rag import BaseRAG
   ```

3. **Register in CLI** (see Step 2 above)

4. **Register in Platform** (see Step 3 above)

5. **Add dependencies** to main `pyproject.toml`

6. **Run tests:**
   ```bash
   uv run pytest
   uv run rag-eval prepare --rag-type my_custom_rag --input-dir data/raw
   uv run rag-eval evaluate --rag-type my_custom_rag
   ```

---

## Complete Integration Checklist

This is the comprehensive list of ALL files that need updates when adding a new RAG type.

### Critical Files (Required)

| File | What to Add |
|------|-------------|
| `src/rag_evaluator/rag_implementations/<name>/` | Your RAG implementation directory |
| `src/rag_evaluator/cli.py` | 1. Import statement<br>2. `elif` in `get_rag_implementation()`<br>3. Update `choices` in argparse (2 places) |
| `platform/backend/app/services/rag_registry.py` | `RAGTypeInfo` in `get_rag_types()` |
| `platform/backend/app/services/rag_adapter.py` | 1. Entry in `RAG_TYPE_REGISTRY`<br>2. Entry in `RAG_TYPE_PARAMETERS`<br>3. Parameter mapping in `create_rag_from_config()`<br>4. Parameter mapping in `create_rag_for_index()` |
| `platform/backend/app/services/index_build_service.py` | Entry in `RAG_TYPE_TO_STORAGE` mapping |

### Frontend Files (Usually Auto-Handled)

| File | What to Add | Notes |
|------|-------------|-------|
| `platform/frontend/src/components/rag-configs/RAGConfigList.tsx` | Icon in `getTypeIcon()` switch | Optional, falls back to default icon |
| `platform/frontend/src/components/rag-configs/RAGConfigDialog.tsx` | Nothing | Auto-fetches from API |

### Test Files (Recommended)

| File | What to Add |
|------|-------------|
| `tests/test_<your_rag>.py` | Unit tests for your RAG implementation |
| `platform/backend/tests/test_api/test_rag_configs.py` | Test cases for new type's parameters |
| `platform/backend/tests/test_services/test_knowledge_base_indexing.py` | Integration test with your RAG |

### Documentation (Recommended)

| File | What to Add |
|------|-------------|
| `README.md` | Add to feature list (line ~28-31) |
| `docs/rag_strategies.md` | Full section describing your RAG approach |
| `docs/api.md` | Update if adding new endpoints |

### Database Considerations

- **No migration needed** for adding a new RAG type - the `rag_type` field is a simple string
- Migration only needed if you add new columns to models

### Storage Type Mapping

In `index_build_service.py`, map your RAG to a storage backend:

```python
RAG_TYPE_TO_STORAGE: dict[str, str] = {
    "vector_semantic": "chroma",
    "vector_hybrid": "qdrant",
    "graph_rag": "neo4j",
    "filesystem_rag": "filesystem",
    "my_custom_rag": "custom",  # <-- Add your storage type
}
```

This determines how index storage is isolated. Use an existing type if your RAG uses the same backend (e.g., `"chroma"` if you use ChromaDB), or create a new identifier.

---

## Testing Your Implementation

### Unit Tests

Create tests in `tests/test_my_custom_rag.py`:

```python
import pytest
from rag_evaluator.rag_implementations.my_custom_rag import MyCustomRAG


class TestMyCustomRAG:
    def test_initialization(self):
        rag = MyCustomRAG()
        assert rag.name == "My Custom RAG"

    def test_prepare_documents(self, tmp_path):
        # Create test documents
        doc = tmp_path / "test.txt"
        doc.write_text("This is test content for RAG indexing.")

        rag = MyCustomRAG()
        rag.prepare_documents(str(tmp_path))

        metrics = rag.get_metrics()
        assert metrics["total_chunks"] > 0

    def test_query(self, tmp_path):
        # Setup
        doc = tmp_path / "test.txt"
        doc.write_text("Paris is the capital of France.")

        rag = MyCustomRAG()
        rag.prepare_documents(str(tmp_path))

        # Test query
        result = rag.query("What is the capital of France?")

        assert "answer" in result
        assert "context" in result
        assert len(result["context"]) > 0
```

### Integration Test with Evaluation

```bash
# Prepare test documents
uv run rag-eval prepare --rag-type my_custom_rag --input-dir data/raw

# Run evaluation
uv run rag-eval evaluate --rag-type my_custom_rag --verbose

# Compare with other implementations
uv run rag-eval evaluate --rag-type my_custom_rag --combine
```

### Backend API Tests

Add tests for your RAG type's parameter validation in `platform/backend/tests/test_api/test_rag_configs.py`:

```python
def test_get_my_custom_rag_parameters(client: TestClient):
    """Test that my_custom_rag parameters are returned correctly."""
    response = client.get("/api/v1/rag-configs/rag-types/my_custom_rag/parameters")
    assert response.status_code == 200

    data = response.json()
    param_names = [p["name"] for p in data["parameters"]]
    assert "chunk_size" in param_names
    assert "search_strategy" in param_names


def test_create_rag_config_with_custom_type(
    client: TestClient,
    sample_project: Project
):
    """Test creating a RAG config with custom type."""
    response = client.post(
        f"/api/v1/projects/{sample_project.id}/rag-configs",
        json={
            "name": "Test Custom RAG",
            "rag_type": "my_custom_rag",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "parameters": {
                "chunk_size": 500,
                "search_strategy": "depth_first",
            },
        },
    )
    assert response.status_code == 201
    assert response.json()["rag_type"] == "my_custom_rag"
```

---

## Linting and Code Quality

Before submitting your RAG implementation, ensure it passes all quality checks.

### Core RAG Implementation

```bash
# Format code
uv run ruff format src/rag_evaluator/rag_implementations/my_custom_rag/

# Lint code
uv run ruff check src/rag_evaluator/rag_implementations/my_custom_rag/

# Type checking
uv run mypy src/rag_evaluator/rag_implementations/my_custom_rag/

# Run all checks
make lint
```

### Backend Changes

```bash
cd platform/backend

# Lint
uv run ruff check app/services/rag_registry.py app/services/rag_adapter.py

# Type check
uv run mypy app/services/

# Run tests
uv run pytest tests/test_api/test_rag_configs.py -v
```

### Frontend Changes (if any)

```bash
cd platform/frontend

# Lint
npm run lint

# Type check
npm run type-check
```

### Common Linting Issues

1. **Missing type annotations:**
   ```python
   # Bad
   def query(self, question, top_k=5):

   # Good
   def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
   ```

2. **Import ordering:**
   ```python
   # Ruff will auto-fix with: uv run ruff check --fix

   # Standard library
   import os
   from pathlib import Path

   # Third-party
   from openai import OpenAI

   # Local
   from rag_evaluator.common.base_rag import BaseRAG
   ```

3. **Line length (100 chars max):**
   ```python
   # Bad - too long
   result = some_function(very_long_parameter_name, another_long_parameter, yet_another_one)

   # Good - split across lines
   result = some_function(
       very_long_parameter_name,
       another_long_parameter,
       yet_another_one,
   )
   ```

4. **Docstrings required for public methods:**
   ```python
   def prepare_documents(self, documents_path: str) -> None:
       """Prepare and index documents for retrieval.

       Args:
           documents_path: Path to directory containing documents.
       """
       pass
   ```

### Pre-Commit Checklist

Before committing, run:

```bash
# From project root
make check  # Runs format → lint → test

# Or run checks in parallel (faster)
make check-parallel -j3
```

---

## Platform UI Integration

To fully integrate with the web platform UI:

### 1. Add to RAG Config Form

Update `platform/frontend/src/components/` to include your RAG type in dropdown menus.

### 2. Add Parameter Form Fields

If your RAG has custom parameters, the `RAG_TYPE_PARAMETERS` schema (from Step 3) will automatically generate form fields.

### 3. Add Documentation

Add a section to `docs/rag_strategies.md` describing your RAG approach.

---

## Best Practices

### 1. Handle Document Loading

Use the built-in document loaders:

```python
from rag_evaluator.common.document_loaders import create_loader

for file_path in documents_dir.glob("*"):
    loader = create_loader(str(file_path))
    if loader:
        content = loader.load()
        # Process content...
```

### 2. Report Progress

For long-running preparation:

```python
def prepare_documents(self, documents_path: str) -> None:
    documents = list(Path(documents_path).glob("*"))
    total = len(documents)

    for i, doc in enumerate(documents):
        # Process document...
        self._report_progress(i + 1, total)  # Built-in progress callback
```

### 3. Thread Safety

The base class provides thread-safe token tracking. If you add shared state, use locks:

```python
def __init__(self, ...):
    super().__init__(...)
    self._my_lock = threading.Lock()

def query(self, ...):
    with self._my_lock:
        # Access shared state safely
```

### 4. Resource Cleanup

Always implement `close()` if you hold resources:

```python
def close(self) -> None:
    if hasattr(self, "_db_connection"):
        self._db_connection.close()
    if hasattr(self, "_client"):
        self._client.shutdown()
```

### 5. Comprehensive Tracing

Use `RetrievalTrace` to provide debugging insight:

```python
trace = RetrievalTrace(strategy="my_strategy")

trace.add_step(
    step_type="embedding",
    input_data={"query": question},
    duration_ms=50.0,
)

trace.add_step(
    step_type="vector_search",
    input_data={"top_k": top_k},
    output_refs=["chunk_1", "chunk_2"],
    duration_ms=100.0,
    metadata={"index_name": "my_index"},
)
```

---

## Summary

### All Integration Points

| Category | File | Changes Required | Priority |
|----------|------|------------------|----------|
| **Implementation** | `src/rag_evaluator/rag_implementations/<name>/` | Create directory with RAG class | Required |
| **CLI** | `src/rag_evaluator/cli.py` | Import + `elif` branch + argparse choices | Required |
| **Registry** | `platform/backend/app/services/rag_registry.py` | `RAGTypeInfo` with parameters | Required |
| **Adapter** | `platform/backend/app/services/rag_adapter.py` | Registry + parameters + instantiation | Required |
| **Storage** | `platform/backend/app/services/index_build_service.py` | `RAG_TYPE_TO_STORAGE` mapping | Required |
| **Frontend Icon** | `platform/frontend/.../RAGConfigList.tsx` | `getTypeIcon()` switch case | Optional |
| **Dependencies** | `pyproject.toml` | Add new packages | If needed |
| **Core Tests** | `tests/test_<your_rag>.py` | Unit tests | Recommended |
| **API Tests** | `platform/backend/tests/test_api/test_rag_configs.py` | Parameter tests | Recommended |
| **README** | `README.md` | Feature list update | Recommended |
| **RAG Strategies** | `docs/rag_strategies.md` | Architecture section | Recommended |

### Minimum Viable Integration

For CLI-only use (no web platform):
1. Create implementation in `src/rag_evaluator/rag_implementations/`
2. Update `cli.py` (import + factory + argparse)

For full platform integration, add:
3. `rag_registry.py` - UI metadata
4. `rag_adapter.py` - Instantiation logic
5. `index_build_service.py` - Storage mapping

### Quick Reference: File Locations

```
RAG-evaluator/
├── src/rag_evaluator/
│   ├── cli.py                          # CLI factory function
│   ├── common/
│   │   ├── base_rag.py                 # BaseRAG interface
│   │   └── provider_interfaces.py      # Data classes
│   └── rag_implementations/
│       └── my_custom_rag/              # YOUR IMPLEMENTATION
│           ├── __init__.py
│           └── my_rag.py
├── platform/backend/app/services/
│   ├── rag_registry.py                 # UI parameter definitions
│   ├── rag_adapter.py                  # Dynamic instantiation
│   └── index_build_service.py          # Storage type mapping
├── platform/frontend/src/components/
│   └── rag-configs/
│       └── RAGConfigList.tsx           # Icon mapping (optional)
└── tests/
    └── test_my_custom_rag.py           # Your tests
```

---

## Need Help?

- Check existing implementations in `src/rag_evaluator/rag_implementations/` for examples
- Review the [RAG Strategies Guide](rag_strategies.md) for architecture details
- See the [API Reference](api.md) for platform integration details
- Use the [custom-rag-template](../custom-rag-template/) for standalone development
