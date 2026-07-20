# RAG Evaluator Documentation

This documentation explains how to install, operate, extend, and deploy the RAG
Evaluator Platform.

## Start Here

| Document | Use it for |
| --- | --- |
| [Getting Started](guides/getting-started.md) | First local setup and first evaluation |
| [Configuration](guides/configuration.md) | Environment variables and provider settings |
| [UI Guide](guides/ui-guide.md) | Web application workflow |
| [CLI Reference](cli.md) | Local preparation, evaluation, and reports |
| [Troubleshooting](guides/troubleshooting.md) | Common setup, indexing, and evaluation issues |

## Core Concepts

| Document | Use it for |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | System layout, data flow, and component boundaries |
| [RAG Strategies](rag_strategies.md) | Built-in RAG implementations and when to use each one |
| [Filesystem RAG internals](../src/rag_evaluator/rag_implementations/filesystem_rag/FILESYSTEM_RAG.md) | Filesystem RAG indexing, BM25 prefetch, agent tools, and traces |
| [RLM-RAG internals](../src/rag_evaluator/rag_implementations/rlm_rag/RLM_RAG.md) | RLM preparation, simple/agent query modes, generated Python, and limitations |
| [Metrics](metrics.md) | Faithfulness, relevancy, precision, recall, and G-Eval correctness |
| [Evaluation Guide](guides/evaluation-guide.md) | Test design, metric selection, and result interpretation |

## Reference

| Document | Use it for |
| --- | --- |
| [API Reference](api.md) | REST endpoints exposed by the FastAPI backend |
| [Custom RAG Integration](custom_rag_integration.md) | Adding a new RAG implementation |
| [Deployment](deployment.md) | Docker, production, and operational guidance |
| [Security](guides/security.md) | Authentication, secrets, and data handling recommendations |

## Common Workflows

### Evaluate Documents In The Web UI

1. Follow [Getting Started](guides/getting-started.md).
2. Create a project.
3. Upload documents into a knowledge base.
4. Create a RAG configuration.
5. Build an index.
6. Create or import a test set.
7. Run an evaluation and inspect results.

### Compare RAG Strategies

1. Build one index per RAG strategy for the same knowledge base.
2. Run each evaluation against the same test set.
3. Use the project comparison tab to choose a baseline and compared evaluations.
4. Review aggregate deltas, per-question differences, cost, and latency.

### Use The CLI

1. Add documents under `data/raw`.
2. Create a JSON test set with a top-level `test_cases` array.
3. Run `rag-eval prepare`.
4. Run `rag-eval evaluate`.
5. Open the generated reports in `reports/` or launch `rag-eval ui`.

### Add A New RAG

1. Implement `BaseRAG`.
2. Register the class and parameter schema in the shared RAG registry.
3. Update backend parameter metadata if the web UI should expose it.
4. Add tests and documentation.

## Repository Links

- [README](../README.md)
- [Contributing](../CONTRIBUTING.md)
- [License](../LICENSE)
- [GitHub issues](https://github.com/fabrizioamort/RAG-evaluator/issues)
