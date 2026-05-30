# CLI Reference

The `rag-eval` CLI runs local document preparation, evaluation, report generation,
and the legacy Streamlit report viewer. It uses the core package in `src/rag_evaluator`
and the root `.env` file.

## Install

Run from the repository root:

```powershell
uv sync --all-extras
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY
```

Check the CLI:

```powershell
uv run rag-eval --help
```

## Test Set Format

CLI evaluations require a JSON object with a top-level `test_cases` array.

```json
{
  "metadata": {
    "name": "Example test set"
  },
  "test_cases": [
    {
      "question": "What does the system evaluate?",
      "expected_answer": "It evaluates Retrieval Augmented Generation systems.",
      "ground_truth_context": [
        "RAG Evaluator compares Retrieval Augmented Generation systems."
      ],
      "difficulty": "easy",
      "category": "overview"
    }
  ]
}
```

`question` and `expected_answer` are required. `ground_truth_context` is optional but
recommended for contextual precision and recall.

## Commands

### `prepare`

Indexes documents for a RAG implementation.

```powershell
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw
```

Options:

| Option | Description | Default |
| --- | --- | --- |
| `--rag-type` | One of `vector_semantic`, `vector_hybrid`, `graph_rag`, `filesystem_rag`, `rlm_rag` | `vector_semantic` |
| `--input-dir` | Directory containing source documents | `data/raw` |

Supported source formats are handled by the shared document loaders and include PDF,
DOCX, TXT, and Markdown where supported by the implementation.

### `evaluate`

Runs DeepEval metrics against a prepared RAG implementation.

```powershell
uv run rag-eval evaluate --rag-type vector_semantic --test-set data/test_set.json
```

Options:

| Option | Description | Default |
| --- | --- | --- |
| `--rag-type` | RAG type to evaluate, or `all` | `vector_semantic` |
| `--test-set` | JSON test set path | `data/test_set.json` |
| `--output` | Report output directory | `reports` |
| `--verbose` | Print per-case progress and errors | disabled |
| `--combine` | Combine this run with latest reports for other RAG types | disabled |

Reports are written as JSON and Markdown. Running `--rag-type all` creates a comparison
report across every registered RAG implementation. Running a single implementation with
`--combine` loads the latest compatible reports from `--output` for the other RAG types
and produces a comparison report.

### `ui`

Starts the legacy Streamlit report viewer for local report inspection.

```powershell
uv run rag-eval ui
```

The production web platform lives under `platform/` and is started separately. See the
[Getting Started guide](guides/getting-started.md).

## RAG Type Notes

| RAG type | Required services | Notes |
| --- | --- | --- |
| `vector_semantic` | None beyond OpenAI-compatible API access | Uses ChromaDB persistence under `CHROMA_PERSIST_DIRECTORY`. |
| `vector_hybrid` | Qdrant | Start with `docker-compose up -d qdrant`. Uses dense embeddings and SPLADE sparse vectors. |
| `graph_rag` | Neo4j | Start with `docker-compose up -d neo4j`. Graph construction uses LLM calls during preparation. |
| `filesystem_rag` | None beyond LLM access | Builds a prepared filesystem and uses an agentic navigation loop. |
| `rlm_rag` | None beyond LLM access | Builds a prepared filesystem and lets a recursive language-model agent explore it with Python tools. |

## Environment Variables

The CLI reads the repository root `.env`.

Common settings:

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | Required for OpenAI models and embeddings. |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible endpoint, such as OpenRouter. |
| `OPENAI_MODEL` | Generation and judge model used by the core CLI. |
| `EMBEDDING_MODEL` | Embedding model for vector implementations. |
| `CHROMA_PERSIST_DIRECTORY` | Local ChromaDB path. |
| `QDRANT_URL` | Qdrant endpoint for hybrid search. |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | Neo4j connection details for Graph RAG. |
| `DEEPEVAL_ASYNC_MODE` | Set to `False` for conservative sequential judging. |

See [Configuration](guides/configuration.md) for the full list.

## Legal RAG Bench Converter

The repository includes a helper for converting Legal RAG Bench JSONL data into CLI
inputs.

Place the source files here:

```text
data/datasets/legal-rag-bench/
  corpus.jsonl
  qa.jsonl
```

Run:

```powershell
uv run python scripts/convert_legal_rag_bench.py
```

Outputs:

```text
data/legal_rag_bench/
  subset/
    raw/
    test_set.json
  full/
    raw/
    test_set.json
```

Smoke test:

```powershell
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/legal_rag_bench/subset/raw
uv run rag-eval evaluate --rag-type vector_semantic --test-set data/legal_rag_bench/subset/test_set.json --verbose
```

## Examples

```powershell
# Prepare and evaluate semantic search
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw
uv run rag-eval evaluate --rag-type vector_semantic

# Compare all registered implementations
uv run rag-eval evaluate --rag-type all --test-set data/test_set.json

# Evaluate one implementation and combine with latest reports for the others
uv run rag-eval evaluate --rag-type rlm_rag --combine

# Use an OpenAI-compatible provider
$env:OPENAI_BASE_URL="https://openrouter.ai/api/v1"
uv run rag-eval evaluate --rag-type vector_semantic
```
