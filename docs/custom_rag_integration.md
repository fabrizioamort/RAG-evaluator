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

The backend adapter and RAG type API use the shared registry to instantiate RAG classes
and expose UI metadata. Keep the core registry as the source of truth for display names,
descriptions, parameter schemas, lifecycle phase, and platform-managed flags.

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

- `load_index()` to attach to an existing index without rebuilding or mutating
  artifacts. The platform uses this when querying a ready index.
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
                "phase": "query",
            },
        },
    },
}
```

Every exposed parameter must declare `phase`:

- `build` for values that affect physical artifacts or preparation.
- `query` for values that can safely change when querying a ready index.

Set `platform_managed: True` on build-time storage fields that the platform should fill
with isolated per-index values.

After this change, the CLI accepts the new type:

```powershell
uv run rag-eval prepare --rag-type my_rag --input-dir data/raw
uv run rag-eval evaluate --rag-type my_rag --test-set data/test_set.json
```

## Platform UI Metadata

The web UI calls backend endpoints such as `/api/v1/rag-types`. Those endpoints adapt
the core registry metadata into API schemas, so adding the core registry entry is enough
for the UI to discover the RAG type. Supported parameter types are `string`, `integer`,
`float`, and `boolean`. For strings, add `choices=[...]` to render a select control.

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

Query-time construction uses the effective config snapshot. Build-phase parameters are
loaded from the selected index snapshot, query overrides are applied only to query-phase
parameters, and `top_k` is carried separately to the query call.

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
- Use `RAGConfig.embedding_model` for build-time embedding choices instead of reading
  only global defaults.
- Mark parameter lifecycle correctly in the registry so build artifacts remain
  reproducible and query overrides are validated.
- Store platform index data under the index storage path.
- Implement `load_index()` when the implementation needs explicit setup before querying
  an existing prepared index.
- Implement `close()` for database clients and file handles.
- Report token usage so cost tracking is meaningful.
- Start with a small corpus and test set before running full evaluations.
- Document the strategy in [RAG Strategies](rag_strategies.md) when it becomes generally useful.
