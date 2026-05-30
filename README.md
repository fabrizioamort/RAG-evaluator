# RAG Evaluator Platform

RAG Evaluator is an open source platform for comparing Retrieval Augmented Generation
(RAG) systems on your own documents. It gives you a web UI for experiment management
and a CLI for local or CI workflows, both backed by the same Python RAG and evaluation
engine.

![RAG Evaluator dashboard](docs/images/hero-screenshot.png)

## What It Does

- Organize RAG experiments by project.
- Upload documents and build isolated indexes for different RAG strategies.
- Reuse ready indexes with query-time overrides for generation model, top-k, and
  strategy-specific query controls.
- Create, import, generate, review, and version test sets.
- Run evaluations with DeepEval metrics and LLM judge reasoning.
- Inspect per-question results, retrieval traces, run manifests, latency, tokens, and cost.
- Compare evaluations side by side and track score trends over time.
- Query ready indexes in the playground before running a full evaluation.
- Use the CLI for repeatable local experiments and report generation.

## Supported RAG Strategies

| Key | Strategy | Backend | Best fit |
| --- | --- | --- | --- |
| `vector_semantic` | Dense semantic search | ChromaDB | General Q&A and baseline runs |
| `vector_hybrid` | Dense + sparse hybrid search | Qdrant | Technical content, exact terms, acronyms |
| `graph_rag` | Graph-enhanced retrieval | Neo4j | Relationship and multi-hop questions |
| `filesystem_rag` | Agentic filesystem navigation | Local prepared files | Large document sets and research-style queries |
| `rlm_rag` | Recursive language-model RAG | Local prepared files + Python tools | Large corpora that benefit from programmatic exploration |

## Quick Start

The most reliable development setup is to run the databases with Docker and run the
backend/frontend locally.

```powershell
git clone https://github.com/fabrizioamort/RAG-evaluator.git
cd RAG-evaluator

Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY

uv sync --all-extras
docker-compose up -d postgres qdrant neo4j

cd platform/backend
uv sync --all-extras
uv run python dev_server.py
```

In a second terminal:

```powershell
cd platform/frontend
npm install
npm run dev
```

Open:

- Web UI: <http://localhost:3000>
- API docs: <http://localhost:8000/api/v1/docs>
- Neo4j browser: <http://localhost:7474>
- Qdrant API: <http://localhost:6333>

For a containerized application stack, see [Deployment](docs/deployment.md).

Platform evaluations separate index-build configuration from query-time settings. Build
choices such as embedding model, chunking, sparse model, graph extraction model, and
storage location are captured in the index snapshot. Evaluation and playground runs can
override only query-safe settings such as the generation model, top-k, and query-phase
RAG parameters without rebuilding the index.

## CLI Example

```powershell
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw
uv run rag-eval evaluate --rag-type vector_semantic --test-set data/test_set.json
uv run rag-eval ui
```

The CLI writes JSON and Markdown reports to `reports/`. Test sets use this shape:

```json
{
  "test_cases": [
    {
      "question": "What is RAG?",
      "expected_answer": "Retrieval Augmented Generation combines retrieval with generation.",
      "ground_truth_context": ["Optional source text used by contextual metrics."],
      "difficulty": "easy"
    }
  ]
}
```

## Documentation

- [Documentation index](docs/index.md)
- [Getting started guide](docs/guides/getting-started.md)
- [CLI reference](docs/cli.md)
- [API reference](docs/api.md)
- [RAG strategies](docs/rag_strategies.md)
- [Evaluation metrics](docs/metrics.md)
- [Configuration](docs/guides/configuration.md)
- [Custom RAG integration](docs/custom_rag_integration.md)
- [Deployment](docs/deployment.md)

## Development

Prerequisites:

- Python 3.11+
- Node.js 18+
- `uv`
- Docker and Docker Compose

Common commands:

```powershell
make install
make dev-infra
make dev-backend
make dev-frontend
make test
make lint
```

When running backend commands manually, execute them from `platform/backend` so the
correct virtual environment is used.

## Security Note

The open source platform does not include built-in user authentication. For shared or
production deployments, run it behind an authenticated reverse proxy or private network,
use managed secrets for API keys, and back up the database and `storage/` directory.

## License

RAG Evaluator is released under the [MIT License](LICENSE).
