# RAG Implementation Guide

This guide details the four distinct Retrieval-Augmented Generation (RAG) strategies available in the RAG Evaluator Platform. Each implementation represents a different approach to information retrieval, ranging from standard semantic search to advanced agentic systems.

## RAG Parameters (Platform UI)

When you create a RAG configuration in the web UI, the "RAG Parameters" section shows the fields that are specific to the selected implementation. Defaults are pre-filled and are usually safe to keep. If you leave a parameter blank, the backend will fall back to the corresponding environment setting (for example `.env`) or an internal default.

The platform also manages storage paths and per-index isolation automatically when you build a Knowledge Base index. This means storage-related parameters are typically optional unless you are using the CLI or a custom integration.

## 1. Vector Semantic Search (Baseline)

### Overview

The standard baseline for modern RAG systems. It uses dense vector embeddings to find semantically similar text chunks.

### Architecture

- **Database:** [ChromaDB](https://www.trychroma.com/) (Local persistent vector store)
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dimensions)
- **Chunking:** Recursive Character Splitter (1000 chars, 200 overlap)
- **Search Metric:** Cosine Similarity

### How it Works

1. **Ingestion:** Documents are split into overlapping chunks. Each chunk is converted into a vector embedding using OpenAI's API and stored in ChromaDB.
2. **Retrieval:** The user's question is embedded into the same vector space. The system performs a K-Nearest Neighbors (KNN) search to find the chunks with the highest cosine similarity.
3. **Generation:** The top-k chunks are fed into the LLM as context to generate an answer.

### Best For

- General-purpose Q&A
- Semantic matching (e.g., matching "car" to "vehicle")
- Speed and simplicity

### Configuration Parameters (Platform UI)

| Parameter | Default | How to fill |
| --- | --- | --- |
| `collection_name` | `rag_documents` | Name of the ChromaDB collection. In the platform, indexes use a generated collection name per index, so you can usually leave the default. Set this only if you are reusing an existing collection outside the platform. |
| `persist_directory` | (empty) | Filesystem path for Chroma persistence. The platform stores indexes under `storage/indexes/<index_id>/chroma`, so leave this blank unless you need a custom path for CLI or custom runs. |

---

## 2. Hybrid Search (Dense + Sparse)

### Overview

Combines the semantic understanding of dense vectors with the precise keyword matching of sparse vectors (SPLADE). This overcomes the limitations of dense vectors, which sometimes miss specific terminology or acronyms.

### Architecture

- **Database:** [Qdrant](https://qdrant.tech/)
- **Dense Model:** OpenAI `text-embedding-3-small`
- **Sparse Model:** SPLADE (via FastEmbed)
- **Fusion Algorithm:** Reciprocal Rank Fusion (RRF)

### How it Works

1. **Dual Indexing:** Each document chunk is indexed twice:
   - **Dense Vector:** Captures meaning/concept.
   - **Sparse Vector:** Captures specific keywords and their importance (learned weights).
2. **Parallel Retrieval:** The query is executed against both indexes simultaneously.
3. **RRF Fusion:** The two result sets are merged using Reciprocal Rank Fusion:
   $$score = \frac{1}{k + rank_{dense}} + \frac{1}{k + rank_{sparse}}$$
   This boosts documents that appear highly ranked in both lists.

### Best For

- Technical documentation (specific variable names/error codes)
- Queries requiring both conceptual understanding and exact keyword matches
- Reducing "lost in the middle" phenomena

### Configuration Parameters (Platform UI)

| Parameter | Default | How to fill |
| --- | --- | --- |
| `collection_name` | (empty) | Qdrant collection name. The platform isolates each index with its own collection name, so you can usually leave this blank unless you need to reuse an existing collection. |
| `qdrant_url` | (empty) | Qdrant server URL, for example `http://localhost:6333`. If blank, the backend uses `QDRANT_URL` from `.env` or the core default. |

---

## 3. Graph RAG (Knowledge Graph)

### Overview

Uses a Neo4j knowledge graph to understand relationships between entities. This enables "multi-hop" reasoning where the answer requires connecting disparate pieces of information.

### Architecture

- **Database:** [Neo4j](https://neo4j.com/)
- **Graph Construction:** LLM-based extraction of Nodes (Entities) and Relationships
- **Framework:** `neo4j-graphrag`
- **Retrieval:** Vector Index + Graph Traversal (Cypher)

### How it Works

1. **Graph Construction:** An LLM analyzes documents to extract entities (Person, Org, Concept) and their relationships (MENTIONS, RELATED_TO).
2. **Retrieval:**
   - **Vector Search:** Finds entry point nodes based on similarity.
   - **Graph Traversal:** Explores the graph neighborhood (1-2 hops) to find connected context that might not share keywords with the query but is semantically relevant via a relationship.
3. **Context Enrichment:** The prompt includes not just the text, but the structured relationships found (e.g., "Alice is CEO of TechCorp").

### Best For

- Complex reasoning tasks
- Questions about relationships or structure (e.g., "How does module A interact with module B?")
- Multi-document synthesis

### Configuration Parameters (Platform UI)

| Parameter | Default | How to fill |
| --- | --- | --- |
| `neo4j_uri` | (empty) | Neo4j connection URI, for example `bolt://localhost:7687`. If blank, the backend uses `NEO4J_URI` from `.env` or the core default. |
| `neo4j_username` | (empty) | Neo4j username. If blank, the backend uses `NEO4J_USERNAME` from `.env` or the core default (`neo4j`). |
| `neo4j_password` | (empty) | Neo4j password. If blank, the backend uses `NEO4J_PASSWORD` from `.env` or the core default (empty). |
| `vector_index_name` | `chunk_embeddings` | Name of the Neo4j vector index. Keep the default unless you are integrating with an existing Neo4j index. |

---

## 4. Filesystem RAG (Agentic)

> **Credit:** Inspired by **Izzy Fuller** and their article *[Convergent Evolution in AI Augmented Development](https://dev.to/izzyfuller/convergent-evolution-in-ai-augmented-development-part-2-when-you-build-solutions-before-you-have-2l0o)*.

### Overview

A unique, agentic approach that navigates a document corpus like a human researcher. Instead of retrieving chunks, it treats the dataset as a filesystem that an LLM agent explores, reads, and summarizes.

### Architecture

- **Agent:** ReAct (Reason-Act) Loop
- **Structure:** Hierarchical "virtual" filesystem
- **Tools:** `ls`, `read_file`, `grep`, `find`

### The "Universal Interface" Pattern

The raw documents are transformed into a structured, traversable format optimized for LLM agents:

1. **_meta/**: Entry points and high-level navigation guides.
2. **_index/**: Specialized indexes (Topics, Entities, Timeline) created by clustering content.
3. **_summaries/**: High-level abstracts of every document.
4. **documents/**: The raw content, line-numbered for precise citation.

### How it Works

The agent receives a question and autonomously decides how to answer it:

1. **Plan:** "I need to find X. I'll check the topic index for 'Deployment'."
2. **Navigate:** Uses `list_directory` to explore related folders.
3. **Filter:** Reads summaries to verify relevance without loading massive files.
4. **Read:** Opens specific files (or line ranges) to get the exact answer.
5. **Synthesize:** Composes the final answer based on its research journey.

### Best For

- "Needle in a haystack" problems
- Broad research questions ("Give me an overview of X")
- When context window usage needs to be minimized (agent selects only what it needs)

### Configuration Parameters (Platform UI)

| Parameter | Default | How to fill |
| --- | --- | --- |
| `llm_model` | `gpt-4o-mini` | Model used by the agent for navigation and analysis. In the platform, this is driven by the LLM Settings section, so you can keep this default. |
| `prepared_path` | `data/prepared/filesystem_rag` | Path to the prepared filesystem output. The platform stores this under `storage/indexes/<index_id>/filesystem_rag`, so you can leave the default unless you need a custom path for CLI or custom runs. |
| `word_threshold` | `1000` | Word count threshold for LLM analysis vs heuristic analysis. Lower values increase LLM usage (higher cost), higher values favor heuristics (lower cost). |
| `max_iterations` | `10` | Maximum ReAct loop iterations per query. Increase for deeper exploration, decrease for faster responses. |
| `max_tool_calls` | `20` | Maximum tool calls per query. Increase if the agent needs more steps to find context. |
| `max_file_reads` | `10` | Maximum file reads per query. Increase if answers often require reading many files. |
