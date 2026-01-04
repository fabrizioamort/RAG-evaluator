# Phase 3 Implementation Plan: Graph RAG with neo4j-graphrag

**Goal:** Implement a Graph RAG system using the official `neo4j-graphrag` Python package and integrate it into the `rag-evaluator` framework for comparison against Vector and Hybrid approaches.

## 1. Description and Requirements

The user requested to implement Graph RAG using the `neo4j-graphrag` package.
**Key Constraints:**

- Use existing Neo4j instance (no Docker setup needed).
- Use dynamic schema inference (no pre-defined schema).
- Use LLM model from `.env` file (no hardcoded models).

## 2. Environment Setup & Dependencies

### 2.1 Connection Configuration

- **Task:** Ensure the application can connect to the existing Neo4j instance.
- **Details:**
  - Verify `.env` file contains `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`.
  - Ensure the `neo4j` driver can connect using these credentials.

### 2.2 Python Dependencies

- **Task:** Update `pyproject.toml` to include `neo4j-graphrag` and related dependencies.
- **Dependencies:**
  - `neo4j-graphrag`
  - `neo4j` (driver)
  - `langchain-openai` (for embeddings/LLM integration).
- **Action:** Run `uv add neo4j-graphrag neo4j langchain-openai` to update the lock file.

## 3. Graph Construction Pipeline

The core of Graph RAG is building the graph (nodes and relationships) from the unstructured text.

### 3.1 Schema Inference

- **Task:** Configure `neo4j-graphrag` to infer the schema from document contents.
- **Details:**
  - Instead of defining rigid node labels (e.g., Person, Organization), use the library's capabilities to extract entities and relationships dynamically based on the text.
  - This allows the graph structure to adapt to the specific domain of the documents without manual schema engineering.

### 3.2 Ingestion Implementation

- **Task:** Implement the indexing logic in `src/rag_evaluator/rag_implementations/graph_rag/indexer.py`.
- **Steps:**
    1. **Load Documents:** Use existing document loaders.
    2. **Chunking:** Use `SemanticChunker` or a larger fixed-size chunker (e.g., 2000 chars) to preserve context for relationship extraction.
    3. **Extraction:** Use `neo4j-graphrag`'s extraction pipeline.
       - **CRITICAL:** Configure the LLM using the model name/deployment specified in the `.env` file (e.g., `OPENAI_MODEL_NAME`). Do **NOT** hardcode `GPT-4o-mini`.
       - Allow the extractor to determine appropriate node labels and relationship types.
    4. **Graph Write:** Write the extracted graph to the Neo4j instance.
    5. **Vector Indexing:** Create vector indexes on `Chunk` or `Entity` nodes for hybrid retrieval capabilities.

## 4. RAG Implementation

### 4.1 GraphRAG Class

- **Task:** Create `src/rag_evaluator/rag_implementations/graph_rag/implementation.py`.
- **Details:**
  - Inherit from `BaseRAG`.
  - Initialize the `neo4j-graphrag` retriever using the connection details from `.env`.

### 4.2 Retrieval Strategies

- **Task:** Configure the retriever.
- **Strategy:** Use the **Hybrid Retriever** capabilities of `neo4j-graphrag`:
    1. **Vector Search:** Find relevant chunks/entities by embedding similarity.
    2. **Graph Traversal:** Expand from found nodes to neighbors (1-2 hops) to gather context.
    3. **Context Assembly:** Format the traversed graph result (triplets or text) for the LLM.

## 5. Evaluation & Testing

### 5.1 Unit & Integration Tests

- **Task:** Write tests in `tests/unit/test_graph_rag.py` and `tests/integration/test_graph_rag_flow.py`.
- **Scope:**
  - Verify connection to the existing Neo4j instance.
  - Verify data ingestion (nodes created dynamically).
  - Verify retrieval returns results.

### 5.2 Multi-Hop Test Cases

- **Task:** Create specific test cases that require connecting information from distinct parts of the graph.
- **Details:** Add these to `data/test_set.json` (or a dedicated `graph_test_set.json`).

### 5.3 Running Evaluation

- **Task:** Run the full evaluation pipeline using `rag-eval evaluate --method graph_rag`.
- **Metrics:** Compare specifically on `DeepEval` metrics against the baseline Vector RAG.

## 6. Documentation & Wrap-up

- **Task:** Update `README.md` with instructions on:
  - Configuring Neo4j credentials in `.env`.
  - Running the ingestion pipeline.
- **Deliverable:** Working Graph RAG implementation with performance reports.

## Execution Checklist

- [ ] **Infrastructure**
  - [ ] Update `pyproject.toml`
  - [ ] Verify `.env` configuration
- [ ] **Ingestion**
  - [ ] Create `GraphIndexer` class
  - [ ] Implement dynamic schema extraction pipeline
- [ ] **Retrieval**
  - [ ] Create `GraphRAG` class (BaseRAG implementation)
  - [ ] Configure `neo4j-graphrag` retriever
- [ ] **Validation**
  - [ ] Verify graph creation and schema in Neo4j Browser
  - [ ] Run evaluation
