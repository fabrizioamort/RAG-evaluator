# RAG Evaluator - Context Guide

## Project Overview

**RAG Evaluator** is a comprehensive Python-based framework designed to implement, evaluate, and compare different Retrieval-Augmented Generation (RAG) methodologies. It uses [DeepEval](https://github.com/confident-ai/deepeval) for metrics-driven evaluation (Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall).

### Key Methodologies
1.  **Vector Semantic Search:** Uses **ChromaDB** for standard dense vector retrieval. (Status: ✅ Complete)
2.  **Hybrid Search:** Uses **Qdrant** to combine dense (semantic) and sparse (keyword/SPLADE) vectors with Reciprocal Rank Fusion (RRF). (Status: ✅ Complete)
3.  **Graph RAG:** Uses **Neo4j** and `neo4j-graphrag` for hybrid vector + graph traversal retrieval, enabling dynamic schema inference and multi-hop reasoning. (Status: ✅ Complete)
4.  **Filesystem RAG:** Uses an **LLM-guided Agent** to navigate a prepared filesystem (Markdown + JSON indexes) using tools like `grep`, `find`, and `read_file`. This mimics a developer exploring a codebase. (Status: ✅ Complete)
...
### Key Files for Context
*   `README.md`: Primary documentation.
*   `SPEC.md`: Original technical specification.
*   `PHASE4_PLAN.md`: Detailed implementation plan for Filesystem RAG.
*   `src/rag_evaluator/rag_implementations/filesystem_rag/FILESYSTEM_RAG.md`: In-depth documentation on the agentic retrieval logic.
*   `PHASE3_IMPLEMENTATION_SUMMARY.md`: Summary of Graph RAG implementation.

*   `src/rag_evaluator/common/base_rag.py`: The interface all RAG implementations must follow.
*   `data/test_set.json`: Structure of the evaluation dataset.

## Conventions
*   **Code Style:** Strict adherence to `ruff` and `mypy` standards.
*   **Typing:** All code must be fully type-hinted.
*   **Testing:** High coverage expected; use `pytest` with mocks for external services (Neo4j, OpenAI) in unit tests.
*   **Async:** DeepEval and many RAG operations are async; handle `asyncio` carefully.