# RAG Evaluator

[![Tests](https://github.com/fabrizioamort/RAG-evaluator/workflows/Tests/badge.svg)](https://github.com/fabrizioamort/RAG-evaluator/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://github.com/python/mypy)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://github.com/pytest-dev/pytest)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A comprehensive evaluation framework for comparing different RAG (Retrieval Augmented Generation) methodologies and technologies.

## Overview

The project has evolved into two main components:

1. **Core RAG Implementation CLI**: The original command-line interface for RAG evaluation.
2. **RAG Evaluation Platform (Web UI)**: A new, comprehensive web-based platform for managing projects, knowledge bases, and evaluations.

3. **Vector Semantic Search** ✅ - Using ChromaDB for pure semantic similarity
4. **Hybrid Search** ✅ - Combining dense (semantic) and sparse (keyword) vectors using Qdrant with RRF fusion
5. **Graph RAG** ✅ - Using Neo4j graph database with neo4j-graphrag for hybrid vector + graph retrieval
6. **Filesystem RAG** ✅ - Direct filesystem search with LLM-guided retrieval (Agentic)

## Evaluation Metrics

...

## Quick Start

### Using the CLI

```bash
# Prepare documents for RAG implementations
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw

# Or prepare for Hybrid Search RAG (requires Qdrant - see below)
uv run rag-eval prepare --rag-type vector_hybrid --input-dir data/raw

# Or prepare for Graph RAG (requires Neo4j - see below)
uv run rag-eval prepare --rag-type graph_rag --input-dir data/raw

# Or prepare for Filesystem RAG
uv run rag-eval prepare --rag-type filesystem_rag --input-dir data/raw

# Run evaluation on a specific RAG implementation
uv run rag-eval evaluate --rag-type vector_semantic

# Run evaluation on Hybrid Search RAG
uv run rag-eval evaluate --rag-type vector_hybrid

# Run evaluation on Graph RAG
uv run rag-eval evaluate --rag-type graph_rag

# Run evaluation on Filesystem RAG
uv run rag-eval evaluate --rag-type filesystem_rag

# Run evaluation on all implementations
uv run rag-eval evaluate --rag-type all --output reports

# Launch the web UI
uv run rag-eval ui
```

...

## Filesystem RAG Setup (Agentic)

The Filesystem RAG implementation employs an **LLM-guided agent** that navigates a prepared filesystem structure to find and retrieve relevant information, mimicking how a human developer explores a codebase.

### How Filesystem RAG Works

Unlike traditional RAG that uses vector similarity, Filesystem RAG operates in two stages:

1. **Document Preparation**:
   - Converts all documents (PDF, DOCX, TXT) to clean Markdown.
   - Performs hybrid analysis (Heuristic for simple docs, LLM for complex ones).
   - Builds a structured directory index (`_index/`, `_summaries/`, `_meta/`).
   - Generates topic maps, entity registries, and question seeds.

2. **Agentic Retrieval**:
   - A ReAct-based agent receives the query and routes it (Known-item vs Exploratory).
   - The agent uses tools to navigate the prepared filesystem.
   - It reads summaries before full documents and follows references.

### Using Filesystem RAG

```bash
# 1. Prepare documents (builds the indexed filesystem structure)
uv run rag-eval prepare --rag-type filesystem_rag --input-dir data/raw

# 2. Run evaluation
uv run rag-eval evaluate --rag-type filesystem_rag

# 3. View results and reasoning traces in UI
uv run rag-eval ui
```

### Agent Tools

The agent has access to several specialized tools:

- `list_directory`: Explore the index structure.
- `read_file`: Read document content (supports progressive disclosure for large files).
- `grep_search`: Keyword/Regex searching across the corpus.
- `find_files`: Locate files by name or pattern.
- `get_file_info`: Inspect metadata without reading full content.

### Filesystem RAG Features

- **No Vector DB Required**: Operates directly on the filesystem.
- **Interpretable Reasoning**: Each query generates a "Reasoning Trace" visible in the UI.
- **Progressive Disclosure**: Only reads what is necessary to answer the question.
- **Human-Readable Indexes**: The prepared structure is fully browsable by humans.

### Using the Streamlit UI

The web interface provides interactive visualization of evaluation results with three main tabs:

```bash
# Launch the UI (loads latest evaluation reports)
uv run rag-eval ui

# Alternative way to launch the UI
uv run python scripts/run_streamlit.py
```

**UI Features:**

- **Overview Tab**: Summary statistics, metrics comparison bar charts, accuracy vs latency scatter plots, and key findings
- **Detailed Comparison Tab**: Score distribution histograms, performance by difficulty breakdown, and side-by-side comparison tables
- **Query Explorer Tab**: Filter test cases by difficulty/category/score, view individual question details, and compare implementation responses

The UI automatically loads the most recent evaluation report from the `reports/` directory.

## Hybrid Search RAG Setup (Qdrant)

The Hybrid Search RAG implementation combines **dense vectors** (semantic similarity via OpenAI embeddings) with **sparse vectors** (keyword matching via SPLADE) using Qdrant's native hybrid search with RRF (Reciprocal Rank Fusion).

### Prerequisites

1. **Qdrant Database**: Start Qdrant using Docker Compose:

```bash
# Start Qdrant (from project root)
docker compose up -d qdrant
```

This starts Qdrant on:

- HTTP API: `http://localhost:6333`
- GRPC: `localhost:6334`

1. **Configuration** (optional): Customize in `.env`:

```bash
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=hybrid_rag
HYBRID_CHUNK_SIZE=700
HYBRID_CHUNK_OVERLAP=100
SPARSE_MODEL_NAME=prithvida/Splade_pp_en_v1
```

### How Hybrid Search Works

The Hybrid Search implementation:

1. **Document Indexing**:
   - Loads documents from multiple formats (TXT, PDF, DOCX)
   - Splits into smaller chunks (700 chars) optimized for keyword matching
   - Generates **dense embeddings** using OpenAI `text-embedding-3-small`
   - Generates **sparse embeddings** using SPLADE via FastEmbed
   - Stores both vector types in Qdrant with named vectors

2. **Hybrid Retrieval**:
   - **Dense Search**: Finds semantically similar chunks
   - **Sparse Search**: Finds chunks with matching keywords/terms
   - **RRF Fusion**: Combines both result sets using Reciprocal Rank Fusion

3. **Answer Generation**:
   - Uses retrieved context from hybrid search
   - Generates answers using LLM

### Using Hybrid Search RAG

```bash
# 1. Start Qdrant
docker compose up -d qdrant

# 2. Prepare documents (indexes with both dense and sparse vectors)
uv run rag-eval prepare --rag-type vector_hybrid --input-dir data/raw

# 3. Run evaluation
uv run rag-eval evaluate --rag-type vector_hybrid

# 4. View results in UI
uv run rag-eval ui
```

### Hybrid Search Features

- **Keyword + Semantic**: Best of both worlds - precise term matching and semantic understanding
- **RRF Fusion**: Robust fusion algorithm that doesn't require score normalization
- **Smaller Chunks**: 700-char chunks optimized for keyword matching
- **SPLADE Embeddings**: State-of-the-art sparse embeddings for term importance

## Graph RAG Setup (Neo4j)

The Graph RAG implementation uses **neo4j-graphrag**, the official Neo4j GraphRAG package for Python, to build knowledge graphs from documents and perform hybrid retrieval combining vector search with graph traversal.

### Prerequisites

1. **Neo4j Database**: You need a running Neo4j instance (version 5.x or later)
   - **Local Installation**: Download from [neo4j.com/download](https://neo4j.com/download/)
   - **Neo4j Desktop**: Easiest option for local development
   - **Neo4j Aura**: Free cloud-hosted option
   - **Docker**: `docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:latest`

2. **Configuration**: Set Neo4j credentials in `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### How Graph RAG Works

The Graph RAG implementation:

1. **Document Ingestion**:
   - Loads documents from multiple formats (TXT, PDF, DOCX)
   - Uses LLM to dynamically extract entities and relationships (no pre-defined schema required)
   - Builds a knowledge graph in Neo4j with interconnected entities

2. **Hybrid Retrieval**:
   - **Vector Search**: Finds semantically similar chunks using embeddings
   - **Graph Traversal**: Expands from found nodes to related entities via graph relationships
   - Enriches context with graph structure (entities, relationships)

3. **Answer Generation**:
   - Combines retrieved text chunks with graph metadata
   - Generates answers using LLM with enhanced context

### Using Graph RAG

```bash
# 1. Prepare documents (builds knowledge graph)
uv run rag-eval prepare --rag-type graph_rag --input-dir data/raw

# 2. Run evaluation
uv run rag-eval evaluate --rag-type graph_rag

# 3. View results in UI
uv run rag-eval ui
```

### Graph RAG Features

- **Dynamic Schema Inference**: No need to predefine entity types or relationships
- **Multi-hop Reasoning**: Excels at questions requiring connections across multiple entities
- **Graph-Enhanced Context**: Retrieves not just matching text but related concepts
- **Vector + Graph Hybrid**: Combines semantic similarity with structural relationships

### Viewing the Knowledge Graph

You can explore the generated knowledge graph using Neo4j Browser:

1. Open Neo4j Browser (typically at `http://localhost:7474`)
2. Run queries to explore the graph:

```cypher
// View all node types and counts
MATCH (n)
RETURN labels(n)[0] AS type, count(*) AS count
ORDER BY count DESC

// View entities and their relationships
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 25

// Search for specific entities
MATCH (n)
WHERE n.name CONTAINS 'RAG'
RETURN n
```

## Evaluation Framework

The project includes a comprehensive evaluation pipeline powered by [DeepEval](https://github.com/confident-ai/deepeval).

### Running Evaluations

```bash
# Prepare documents (one-time setup)
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw

# Run evaluation with default test set
uv run rag-eval evaluate --rag-type vector_semantic

# Run evaluation with custom test set
uv run rag-eval evaluate --rag-type vector_semantic --test-set my_tests.json

# Run with verbose output
uv run rag-eval evaluate --rag-type vector_semantic --verbose

# Alternative: use the evaluation script directly
uv run python scripts/run_evaluation.py --rag-type vector_semantic --verbose
```

### Test Dataset

The evaluation uses a test dataset (`data/test_set.json`) with question-answer pairs. Each test case includes:

- **question**: The query to test
- **expected_answer**: The ground truth answer
- **ground_truth_context**: Reference context chunks
- **difficulty**: Test case difficulty (easy/medium/hard)
- **category**: Question type (definition/explanation/comparison/etc.)

Example test case:

```json
{
  "id": "tc_001",
  "question": "What is RAG?",
  "expected_answer": "RAG (Retrieval Augmented Generation) combines...",
  "ground_truth_context": ["RAG is a technique that..."],
  "difficulty": "easy",
  "category": "definition"
}
```

### Evaluation Metrics

The framework evaluates RAG implementations across four key metrics:

1. **Faithfulness** (threshold: 0.7)
   - Measures if the answer is derived only from the retrieved context
   - Prevents hallucination

2. **Answer Relevancy** (threshold: 0.7)
   - Measures if the answer actually addresses the question
   - Ensures responses are on-topic

3. **Contextual Precision** (threshold: 0.7)
   - Measures if the retrieved documents are relevant
   - Evaluates retrieval quality

4. **Contextual Recall** (threshold: 0.7)
   - Measures if all relevant information was retrieved
   - Ensures comprehensive context

### Evaluation Reports

Each evaluation generates comprehensive reports with enhanced statistical analysis:

**JSON Report** (`reports/eval_<impl>_<timestamp>.json`)

- Machine-readable results
- Complete metric details
- Detailed per-test-case results

**Markdown Report** (`reports/eval_<impl>_<timestamp>.md`)

- Human-readable summary with multiple sections:
  - **Metrics Summary**: Overall scores with pass/fail indicators
  - **Statistical Analysis**: Mean, median, std dev, and 95% confidence intervals for each metric
  - **Performance by Difficulty**: Breakdown by easy/medium/hard questions
  - **Failure Analysis**: Detailed analysis of low-scoring test cases
  - **Statistical Comparison**: Pairwise t-tests between implementations (in comparison reports)
  - **Detailed Results**: Individual test case breakdowns with performance data

### Customizing Thresholds

Metric thresholds can be customized in `.env`:

```bash
EVAL_FAITHFULNESS_THRESHOLD=0.8
EVAL_ANSWER_RELEVANCY_THRESHOLD=0.75
EVAL_CONTEXTUAL_PRECISION_THRESHOLD=0.7
EVAL_CONTEXTUAL_RECALL_THRESHOLD=0.7
```

### Comparing Implementations

Compare multiple RAG implementations (coming soon):

```bash
uv run rag-eval evaluate --rag-type all
```

This generates a comparison report highlighting strengths and weaknesses of each approach.

## RAG Evaluation Platform (Web UI)

A comprehensive web-based platform to manage RAG projects, knowledge bases, and evaluations.

### Quick Start (Web UI)

#### Backend Setup

```bash
cd platform/backend
cp .env.example .env  # Edit .env with your keys
uv sync --all-extras
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

#### Frontend Setup

```bash
cd platform/frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:3000`.

## Project Structure

```
RAG-evaluator/
├── src/rag_evaluator/
│   ├── rag_implementations/     # RAG implementation modules
│   │   ├── vector_semantic/     # ChromaDB semantic search
│   │   ├── vector_hybrid/       # Hybrid search
│   │   ├── graph_rag/           # Neo4j graph RAG
│   │   └── filesystem_rag/      # Filesystem-based RAG
│   ├── evaluation/              # Evaluation framework
│   │   ├── evaluator.py         # Main evaluation logic
│   │   ├── report_generator.py  # Enhanced report generation
│   │   ├── statistics.py        # Statistical analysis module
│   │   └── difficulty_analysis.py # Difficulty breakdown analysis
│   ├── common/                  # Shared utilities and base classes
│   │   ├── base_rag.py          # Abstract base class for RAG
│   │   └── document_loaders.py  # Multi-format document loading
│   ├── ui/                      # Streamlit web interface
│   │   └── streamlit_app.py     # 3-tab interactive dashboard
│   ├── config.py               # Configuration management
│   └── cli.py                  # CLI entry point
├── data/
│   ├── raw/                    # Source documents
│   └── processed/              # Processed documents
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── reports/                    # Evaluation reports
└── scripts/                    # Helper scripts
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/rag_evaluator --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_evaluator.py
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/rag_evaluator
```

## Configuration

Key configuration options in `.env`:

**LLM Configuration:**

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_MODEL` - Model for answer generation (default: gpt-4-turbo-preview, supports gpt-5-nano)
- `EMBEDDING_MODEL` - Model for embeddings (default: text-embedding-3-small)
- `OPENAI_TIMEOUT` - API timeout in seconds (default: 600)

**Database Configuration:**

- `CHROMA_PERSIST_DIRECTORY` - ChromaDB storage location (default: ./data/chroma_db)
- `QDRANT_URL` - Qdrant HTTP endpoint (default: <http://localhost:6333>)
- `QDRANT_COLLECTION_NAME` - Qdrant collection name (default: hybrid_rag)
- `NEO4J_URI` - Neo4j connection URI (for Graph RAG)
- `NEO4J_PASSWORD` - Neo4j password

**Hybrid Search Configuration:**

- `HYBRID_CHUNK_SIZE` - Chunk size for hybrid search (default: 700)
- `HYBRID_CHUNK_OVERLAP` - Chunk overlap (default: 100)
- `SPARSE_MODEL_NAME` - FastEmbed sparse model (default: prithvida/Splade_pp_en_v1)

**Evaluation Configuration:**

- `EVAL_TEST_SET_PATH` - Path to test dataset (default: data/test_set.json)
- `EVAL_REPORTS_DIR` - Reports output directory (default: reports)
- `EVAL_FAITHFULNESS_THRESHOLD` - Faithfulness metric threshold (default: 0.7)
- `EVAL_ANSWER_RELEVANCY_THRESHOLD` - Answer relevancy threshold (default: 0.7)
- `EVAL_CONTEXTUAL_PRECISION_THRESHOLD` - Context precision threshold (default: 0.7)
- `EVAL_CONTEXTUAL_RECALL_THRESHOLD` - Context recall threshold (default: 0.7)

**DeepEval Configuration:**

- `DEEPEVAL_ASYNC_MODE` - Enable parallel evaluation (default: False, set to False to avoid rate limits)
- `DEEPEVAL_PER_TASK_TIMEOUT` - Per-task timeout in seconds (default: 900)
- `DEEPEVAL_PER_ATTEMPT_TIMEOUT` - Per-attempt timeout in seconds (default: 300)
- `DEEPEVAL_MAX_RETRIES` - Maximum retry attempts (default: 3)

## Platform Support

**Windows Compatibility:**

- CLI output optimized for Windows console (no emoji characters that cause encoding errors)
- Proper handling of file paths and line endings
- All features tested on Windows 10/11

**Model Compatibility:**

- Supports both standard OpenAI models (gpt-4-turbo, gpt-4o) and newer models (gpt-5-nano)
- Automatic temperature parameter adjustment for models that don't support it (e.g., gpt-5-nano)

**Rate Limiting:**

- Configurable async mode to prevent API rate limit errors
- Automatic retry logic with exponential backoff
- Timeout configuration for long-running evaluations

## Requirements

- Python 3.11+
- OpenAI API key
- Docker (for Qdrant and Neo4j)
- Qdrant (for Hybrid Search RAG) - via Docker Compose
- Neo4j database (for Graph RAG) - via Docker Compose or external instance

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code quality standards and testing requirements
- How to add new RAG implementations
- Pull request process

## License

MIT
