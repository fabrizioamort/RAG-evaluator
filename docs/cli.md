# CLI Reference - RAG Evaluator

The RAG Evaluator CLI is the core tool for running local evaluations, preparing documents, and testing different RAG implementations directly from your terminal.

## Installation

The CLI is part of the core project. Ensure you have the environment set up:

```bash
# Install dependencies
uv sync
```

## Basic Usage

The CLI is accessed via `uv run rag-eval` (defined in `pyproject.toml`).

```bash
uv run rag-eval --help
```

## Commands

### 1. `prepare`

Prepares and indexes documents for a specific RAG implementation.

**Usage:**
```bash
uv run rag-eval prepare --rag-type <type> --input-dir <path>
```

**Arguments:**
- `--rag-type`: The RAG implementation to prepare. Choices: `vector_semantic`, `vector_hybrid`, `graph_rag`, `filesystem_rag`.
- `--input-dir`: Path to the directory containing raw documents (PDF, DOCX, TXT).

**Examples:**
```bash
# Standard Vector Search (ChromaDB)
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw

# Filesystem RAG
uv run rag-eval prepare --rag-type filesystem_rag --input-dir data/raw
```

### 2. `evaluate`

Runs the evaluation pipeline using the DeepEval framework.

**Usage:**
```bash
uv run rag-eval evaluate --rag-type <type> [options]
```

**Arguments:**
- `--rag-type`: The implementation to evaluate. Use `all` to run all available implementations.
- `--test-set`: Path to the test set JSON file (default: `data/test_set.json`).
- `--output`: Directory to save reports (default: `reports`).
- `--verbose`: Enable detailed output.

**Examples:**
```bash
# Evaluate specific implementation
uv run rag-eval evaluate --rag-type vector_semantic

# Evaluate all implementations
uv run rag-eval evaluate --rag-type all

# Use custom test set
uv run rag-eval evaluate --rag-type vector_hybrid --test-set my_tests.json
```

### 3. `ui`

Launches the standalone Streamlit UI for local visualization.

**Usage:**
```bash
uv run rag-eval ui
```

---

## RAG Implementations

### Vector Semantic Search (ChromaDB)
The default implementation using OpenAI embeddings and ChromaDB.
- **Requires:** `OPENAI_API_KEY`
- **Setup:** `uv run rag-eval prepare --rag-type vector_semantic ...`

### Hybrid Search RAG (Qdrant)
Combines dense vectors (semantic) and sparse vectors (keyword/SPLADE) using Reciprocal Rank Fusion (RRF).

**Prerequisites:**
- Qdrant running (`docker-compose up -d qdrant`)
- Configuration in `.env`:
  ```bash
  QDRANT_URL=http://localhost:6333
  QDRANT_COLLECTION_NAME=hybrid_rag
  SPARSE_MODEL_NAME=prithvida/Splade_pp_en_v1
  ```

**How it works:**
1. Splits docs into 700-char chunks.
2. Generates dense embeddings (OpenAI) and sparse embeddings (SPLADE).
3. Performs hybrid search in Qdrant with RRF fusion.

### Graph RAG (Neo4j)
Uses a knowledge graph to enhance retrieval with structural relationships.

**Prerequisites:**
- Neo4j Database (local or cloud)
- Configuration in `.env`:
  ```bash
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=password
  ```

**How it works:**
1. Uses LLM to extract entities and relationships during ingestion.
2. Builds a graph in Neo4j.
3. Retrieves context by traversing the graph from found entities.

### Filesystem RAG (Agentic)
An agentic approach that navigates a prepared directory structure like a human developer.

**How it works:**
1. **Preparation:** Converts docs to Markdown, builds a structured index (`_index/`, `_summaries/`).
2. **Retrieval:** A ReAct agent uses tools (`list_directory`, `read_file`, `grep_search`) to find answers.

**Agent Tools:**
- `list_directory`: Explore the index.
- `read_file`: Read content (supports partial reading).
- `grep_search`: Keyword search.
- `find_files`: Locate files.
