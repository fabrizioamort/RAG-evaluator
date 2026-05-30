# Custom RAG Integration

RAG Evaluator can evaluate any RAG implementation that follows the `BaseRAG`
interface. This guide shows the integration points for both CLI and web platform use.

## How The Integration Works

The core interface lives in:

```text
src/rag_evaluator/common/base_rag.py
```

The shared registry lives in:

```text
src/rag_evaluator/rag_implementations/registry.py
```

The backend web UI metadata lives in:

```text
platform/backend/app/services/rag_registry.py
```

The backend adapter uses the shared registry to instantiate RAG classes from platform
RAG configurations.

## Implement `BaseRAG`

Create a package under `src/rag_evaluator/rag_implementations/`.

```text
src/rag_evaluator/rag_implementations/my_rag/
  __init__.py
  my_rag.py
```

Minimal implementation:

```python
from typing import Any

from rag_evaluator.common.base_rag import BaseRAG, RAGConfig


class MyRAG(BaseRAG):
    def __init__(self, config: RAGConfig | None = None, my_param: str = "default") -> None:
        super().__init__(name="My RAG", config=config)
        self.my_param = my_param
        self._metrics: dict[str, Any] = {}

    def prepare_documents(self, documents_path: str) -> None:
        # Load and index documents here.
        self._metrics = {"total_chunks": 0}

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        return {
            "answer": "Answer text",
            "context": ["Retrieved context chunk"],
            "metadata": {
                "retrieval_time": 0.0,
                "token_usage": self._token_usage.to_dict(),
            },
        }

    def get_metrics(self) -> dict[str, Any]:
        return self._metrics
```

Recommended optional methods:

- `retrieve()` for retrieval-only traces.
- `generate()` for generation from pre-retrieved context.
- `_get_strategy_name()` for cleaner trace labels.
- `close()` for database clients, files, or background resources.

## Register The RAG For CLI And Backend Instantiation

Edit `src/rag_evaluator/rag_implementations/registry.py`.

Add the class path:

```python
_RAG_CLASS_PATHS = {
    # existing entries...
    "my_rag": "rag_evaluator.rag_implementations.my_rag.my_rag.MyRAG",
}
```

Add user-facing metadata:

```python
RAG_TYPES = {
    # existing entries...
    "my_rag": {
        "name": "My RAG",
        "description": "Short description shown in CLI/platform contexts",
    },
}
```

Add a parameter schema:

```python
RAG_TYPE_PARAMETERS = {
    # existing entries...
    "my_rag": {
        "properties": {
            "my_param": {
                "type": "string",
                "default": "default",
                "description": "Example parameter",
            },
        },
    },
}
```

After this change, the CLI accepts the new type:

```powershell
uv run rag-eval prepare --rag-type my_rag --input-dir data/raw
uv run rag-eval evaluate --rag-type my_rag --test-set data/test_set.json
```

## Add Platform UI Metadata

The web UI calls backend endpoints such as `/api/v1/rag-types`. Add a matching
`RAGTypeInfo` entry in `platform/backend/app/services/rag_registry.py`.

```python
RAGTypeInfo(
    name="my_rag",
    display_name="My RAG",
    description="Short description shown in the UI.",
    requires_index=True,
    parameters=[
        RAGTypeParameter(
            name="my_param",
            type="string",
            description="Example parameter.",
            default="default",
        ),
    ],
)
```

Supported parameter types are `string`, `integer`, `float`, and `boolean`. For strings,
add `choices=[...]` to render a select control.

## Map Platform Parameters To Constructor Arguments

Edit `platform/backend/app/services/rag_adapter.py`. Add a branch in both creation paths
if your constructor needs custom keyword arguments.

```python
elif config_model.rag_type == "my_rag":
    kwargs["my_param"] = parameters.get("my_param", "default")
```

For index-specific construction:

```python
elif rag_type == "my_rag":
    params = config_snapshot.get("parameters", {})
    kwargs["my_param"] = params.get("my_param", "default")
```

If your RAG uses platform-managed storage, derive paths from the provided `index_path` or
the index storage path, not from a fixed global directory.

## Add Index Storage Mapping

Edit `platform/backend/app/services/index_build_service.py`.

```python
RAG_TYPE_TO_STORAGE = {
    # existing entries...
    "my_rag": "my_storage_type",
}
```

If your storage needs cleanup on index deletion, extend `_cleanup_storage()` for the new
storage type. If the RAG uses an existing backend, reuse an existing storage type such as
`chroma`, `qdrant`, `neo4j`, or `filesystem`.

## Add Dependencies

Add Python packages to the root `pyproject.toml`. If the backend imports them directly,
also ensure `platform/backend/pyproject.toml` can resolve them through the editable
`rag-evaluator` dependency or explicit backend dependencies.

Then run:

```powershell
uv sync --all-extras

cd platform/backend
uv sync --all-extras
```

## Add Tests

Recommended tests:

| Area | File |
| --- | --- |
| Core RAG behavior | `tests/unit/test_my_rag.py` |
| Registry entry | `tests/unit/test_rag_registry.py` |
| Backend parameter metadata | `platform/backend/tests/test_api/test_rag_configs.py` |
| Backend adapter construction | `platform/backend/tests/test_services/test_rag_adapter.py` |
| Index build behavior | `platform/backend/tests/test_services/test_knowledge_base_indexing.py` |

Useful commands:

```powershell
uv run pytest tests/unit/test_rag_registry.py -q

cd platform/backend
uv run pytest tests/test_api/test_rag_configs.py tests/test_services/test_rag_adapter.py -q
```

## Test Set Expectations

The CLI evaluator expects this JSON shape:

```json
{
  "test_cases": [
    {
      "question": "Question text",
      "expected_answer": "Ground truth answer",
      "ground_truth_context": ["Optional ground truth context"]
    }
  ]
}
```

Your `query()` method should return:

```python
{
    "answer": "...",
    "context": ["..."],
    "metadata": {
        "retrieval_time": 0.0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0},
    },
}
```

The platform can provide richer debugging when your implementation supports
`query_with_trace()` through `retrieve()` and `generate()`.

## Best Practices

- Keep implementation-specific state inside the RAG class.
- Use `RAGConfig.parameters` for user-controlled settings.
- Use `RAGConfig.llm_model`, `llm_provider`, and `llm_base_url` instead of hard-coded model names.
- Store platform index data under the index storage path.
- Implement `close()` for database clients and file handles.
- Report token usage so cost tracking is meaningful.
- Start with a small corpus and test set before running full evaluations.
- Document the strategy in [RAG Strategies](rag_strategies.md) when it becomes generally useful.
