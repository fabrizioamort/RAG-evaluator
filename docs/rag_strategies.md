# RAG Implementation Guide

This guide details the four distinct Retrieval-Augmented Generation (RAG) strategies available in the RAG Evaluator Platform. Each implementation represents a different approach to information retrieval, ranging from standard semantic search to advanced agentic systems.

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
