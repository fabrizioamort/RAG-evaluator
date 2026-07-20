# RAG Strategies

RAG Evaluator includes six built-in retrieval strategies. They all implement the
same `BaseRAG` interface, so they can be evaluated with the same test sets and metrics.

Use this guide to choose an implementation and understand the trade-offs before you run
an evaluation.

## Strategy Summary

| Type | Storage | Retrieval style | Strengths | Watch-outs |
| --- | --- | --- | --- | --- |
| `vector_semantic` | ChromaDB | Dense vector similarity | Fast baseline, simple setup, strong semantic matching | Can miss exact terms, IDs, and acronyms |
| `vector_hybrid` | Qdrant | Dense + sparse vectors with fusion | Good for technical docs and exact terminology | Requires Qdrant and sparse model loading |
| `graph_rag` | Neo4j | Vector entry points plus graph traversal | Useful for relationships and multi-hop reasoning | LLM graph extraction can be slower and more expensive |
| `filesystem_rag` | Local prepared files | BM25 prefetch plus ReAct file navigation | Inspectable lexical index and flexible, evidence-driven source reading | More latency and model-dependent behavior than fixed top-k search |
| `rlm_rag` | Local prepared files | Generated-Python exploration, or direct context for small corpora | Useful for experiments with programmatic corpus exploration | Generated code is not safely sandboxed; several configured limits are not yet enforced |
| `google_vertex_search` | Google Vertex AI Search (Discovery Engine) | Managed search with automatic parsing, chunking, and embedding | Offloads indexing and retrieval infrastructure to a managed Google service | Requires a GCP project, GCS staging bucket, and the `google-vertex` extra |

## Shared Platform Behavior

In the web platform, documents are uploaded to a knowledge base and then built into
one or more indexes. Each index receives an isolated physical identifier, so you can
build several strategies for the same knowledge base and compare them without storage
collisions.

When you create a RAG configuration, the UI reads parameter metadata from the backend.
Each parameter is marked as either build-time or query-time. Build-time parameters are
captured in the index snapshot and require a new index when they change. Query-time
parameters can be overridden when running an evaluation or playground query against a
ready index.

Most storage parameters can be left blank in the web platform because index storage is
managed automatically. Platform-managed parameters are still recorded in snapshots for
reproducibility, but the platform fills them with isolated per-index values. For CLI
runs, storage paths and service URLs come from the root `.env` file.

## Vector Semantic Search

Type key: `vector_semantic`

The semantic baseline uses dense embeddings and ChromaDB. Documents are split into
chunks, embedded, stored locally, and retrieved by vector similarity.

Best for:

- First baseline evaluations.
- General Q&A where semantic similarity is enough.
- Small to medium corpora where setup speed matters.

Platform parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `collection_name` | `rag_documents` | The platform replaces this with a generated per-index collection unless you provide one. |
| `persist_directory` | empty | Leave blank in the platform to use managed index storage. |
| `embedding_model` | `text-embedding-3-small` | Build-time model stored on the RAG configuration and index snapshot. |
| `chunk_size` | `1000` | Build-time text chunk size. |
| `chunk_overlap` | `200` | Build-time overlap between chunks. |

Query-time settings:

- `llm_model` can be overridden per evaluation or playground query.
- `top_k` is passed to query execution instead of stored in the constructor config.

CLI environment:

- `CHROMA_PERSIST_DIRECTORY`
- `EMBEDDING_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` if using an OpenAI-compatible endpoint

## Hybrid Search

Type key: `vector_hybrid`

Hybrid search combines dense vector retrieval with sparse keyword-aware retrieval and
fuses the result sets. This helps when a query contains product codes, error messages,
legal citations, API names, or other exact terms that pure dense retrieval may blur.

Best for:

- Technical documentation.
- Legal or policy corpora with exact references.
- Queries that mix concepts with precise vocabulary.

Platform parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `collection_name` | empty | Leave blank for platform-managed per-index Qdrant collections. |
| `qdrant_url` | empty | Leave blank to use `QDRANT_URL`; set it for a custom Qdrant host. |
| `embedding_model` | `text-embedding-3-small` | Dense embedding model used while building the index. |
| `embedding_dimension` | `1536` | Dense vector dimension; use `3072` with `text-embedding-3-large`. |
| `sparse_model_name` | `prithivida/Splade_PP_en_v1` | Sparse model used while building sparse vectors. |
| `chunk_size` | `500` | Build-time text chunk size for platform indexes. |
| `chunk_overlap` | `50` | Build-time overlap between chunks for platform indexes. |

Query-time settings:

- `llm_model` can be overridden per evaluation or playground query.
- `top_k` is passed to query execution.

CLI environment:

- `QDRANT_URL`
- `QDRANT_COLLECTION_NAME`
- `HYBRID_CHUNK_SIZE`
- `HYBRID_CHUNK_OVERLAP`
- `HYBRID_FUSION_ALPHA`
- `HYBRID_INDEXING_BATCH_SIZE`
- `SPARSE_MODEL_NAME` (default: `prithivida/Splade_PP_en_v1`)

Start Qdrant locally:

```powershell
docker-compose up -d qdrant
```

## Graph RAG

Type key: `graph_rag`

Graph RAG stores document chunks and extracted relationships in Neo4j. Retrieval starts
with semantic search and can enrich context by traversing graph relationships.

Best for:

- Relationship-heavy corpora.
- Questions like "how does X depend on Y?"
- Multi-document synthesis where structure matters.

Platform parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `neo4j_uri` | empty | Leave blank to use `NEO4J_URI`. |
| `neo4j_username` | empty | Leave blank to use `NEO4J_USERNAME`. |
| `neo4j_password` | empty | Leave blank to use `NEO4J_PASSWORD`. |
| `vector_index_name` | `chunk_embeddings` | Keep the default unless you manage a custom Neo4j vector index. |
| `embedding_model` | `text-embedding-3-small` | Embedding model used for graph vector search artifacts. |
| `extraction_model` | RAG config model | Build-time model used for graph extraction. |

Query-time settings:

- `llm_model` can be overridden per evaluation or playground query.
- `top_k` is passed to query execution.

CLI environment:

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `OPENAI_API_KEY`

Start Neo4j locally:

```powershell
docker-compose up -d neo4j
```

Graph construction uses LLM calls during indexing, so start with a small subset when
testing a new corpus.

## Filesystem RAG

Type key: `filesystem_rag`

Filesystem RAG recursively converts TXT, PDF, and DOCX sources to Markdown, analyzes each
document, and builds an inspectable prepared directory. Its primary deterministic search
artifact is a local section-level BM25 index. Topic, entity, question-seed, and timeline
files provide additional browsing paths.

At query time, the system routes the question as a known-item or exploratory search,
prefetches BM25 and question-seed candidates, and gives them to a ReAct-style agent. The
agent can search, grep, list, and selectively read source line ranges before answering.
Retrieval and generation happen in the same loop, and the public `top_k` is ignored.

Best for:

- Larger corpora where a fixed context window is too restrictive.
- Broad or multi-document research questions.
- Exact-term retrieval combined with adaptive source navigation.
- Experiments where a human-readable index and detailed retrieval trace matter.

Prepared structure:

```text
prepared_path/
  _meta/                       # Overview, navigation guide, statistics
  _index/
    passages/bm25.json         # Section-level lexical index
    topics/                    # Topic-to-document navigation
    entities/                  # Entity-to-document navigation
    questions/                 # Generated question-to-document hints
    temporal/                  # Extracted timeline
  _summaries/                  # Per-document navigation summaries
  documents/                   # Converted Markdown + metadata JSON
  _original/                   # Copied sources
```

Documents below `word_threshold` use heuristic analysis; documents at or above it use
LLM analysis, with heuristic fallback on provider or JSON errors. This affects summaries,
topics, entities, question seeds, and the analysis text included in BM25 records.

Platform parameters:

| Parameter | Phase | Default | Notes |
| --- | --- | --- | --- |
| `prepared_path` | Build | managed | The platform assigns an isolated directory under `storage/indexes`. |
| `word_threshold` | Build | `1000` | Lower values cause more documents to use LLM analysis. |
| `llm_model` | Query/top level | `gpt-4o-mini` | Agent model; also used for LLM analysis while building a new index. |
| `max_iterations` | Query | `10` | Maximum model turns in the ReAct loop. |
| `max_tool_calls` | Query | `20` | Maximum tool calls per query. |
| `max_file_reads` | Query | `10` | Maximum uncached file reads per query. |

Build-time changes require a new index. The model and execution budgets can be overridden
for a ready index. Agent paths are model-dependent, so compare quality together with
latency, token usage, files read, and tool counts.

See [Filesystem RAG: indexing and retrieval internals](../src/rag_evaluator/rag_implementations/filesystem_rag/FILESYSTEM_RAG.md)
for source discovery, analysis, every generated artifact, BM25 scoring, deterministic
prefetch, tool semantics, answer safeguards, and trace behavior.

## RLM-RAG

Type key: `rlm_rag`

RLM-RAG is an RLM-inspired experiment that prepares a document catalog, summaries,
section metadata, and a topic map. For corpora above `small_corpus_threshold`, an
orchestrator model writes Python to inspect those artifacts, read or grep documents, and
optionally delegate analysis to a worker model. For corpora at or below the threshold, it
uses a separate simple-context path: the first `top_k` catalog documents are truncated
and sent directly to the orchestrator without relevance ranking.

Best for:

- Research into programmatic filtering, grouping, and iterative corpus exploration.
- Questions where generated code can test several search or aggregation strategies.
- Controlled, trusted experiments comparing agentic retrieval designs.

It is not currently the best choice for untrusted or multi-tenant content. Both `lite`
and `full` agent security modes execute generated Python, and neither runtime is a
hardened sandbox.

Current security modes:

| Mode | Current behavior |
| --- | --- |
| `lite` | Executes code in the application process with persistent variables. Fast, but the configured REPL timeout is not enforced and namespace restrictions are not a security boundary. |
| `full` | Executes each block in a killable subprocess, but the subprocess currently lacks the `fs`, worker-call, and budget objects required for normal exploration. Injection wrapping is defined in code but not wired into agent initialization. Treat this mode as experimental, not complete isolation. |

Platform parameters:

| Parameter | Phase | Default | Notes |
| --- | --- | --- | --- |
| `prepared_path` | Build | managed | Isolated prepared RLM directory. |
| `worker_model` | Build | `gpt-5-nano` | Builds summaries/topics and serves worker sub-calls. |
| `chunk_size` | Build | `1000` | Included in manifest invalidation, but current preparation does not chunk content. |
| `chunk_overlap` | Build | `200` | Included in manifest invalidation, but current preparation does not chunk content. |
| `use_llm_summaries` | Build | `true` | Enables worker-generated summaries. |
| `use_llm_topics` | Build | `true` | Enables worker-generated topics. |
| `max_topics_per_doc` | Build | `5` | Maximum topic labels per document. |
| `orchestrator_model` | Query | RAG config model | Main reasoning, code-generation, and answer model. |
| `security_mode` | Query | `lite` | Selects in-process or experimental subprocess execution. |
| `max_repl_steps` | Query | `15` | Maximum orchestrator exploration turns. |
| `repl_timeout` | Query | `5.0` | Enforced only by the subprocess REPL. |
| `max_file_reads` | Query | `12` | Enforced for helper reads in lite agent mode. |
| `max_read_bytes` | Query | `50000` | Implemented as a character limit, despite the name. |
| `max_read_lines` | Query | `1000` | Configured but not currently enforced. |
| `max_sub_calls` | Query | `8` | Displayed in budget status but not currently enforced. |
| `max_recursion_depth` | Query | `2` | Limits nested entry into worker-call logic. |
| `small_corpus_threshold` | Query | `10` | Chooses simple-context or generated-code agent mode. |

`top_k` affects only simple-context mode and is ignored in agent mode. If `llm_model` is
overridden without an explicit `orchestrator_model`, the platform uses the new generation
model as the effective orchestrator.

See [RLM-RAG: indexing and retrieval internals](../src/rag_evaluator/rag_implementations/rlm_rag/RLM_RAG.md)
for exact source-processing rules, manifest behavior, both query algorithms, filesystem
APIs, worker calls, traces, configuration gaps, and the current security limitations.

## Google Vertex AI Search

Type key: `google_vertex_search`

This strategy delegates indexing, chunking, embedding, and retrieval to a managed
Google Vertex AI Search data store (Discovery Engine). Documents are staged to a GCS
bucket and imported into the data store, which handles parsing, layout-aware chunking,
and ranking automatically.

Best for:

- Teams that want to offload indexing infrastructure to a managed service.
- Evaluating Google's managed retrieval quality against self-hosted strategies.
- Reusing an existing Vertex AI Search data store for evaluation only.

Requires the optional `google-vertex` extra:

```powershell
uv sync --extra google-vertex
```

Platform parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `data_store_id` | empty | Leave blank to auto-generate an isolated data store for this index, or set it (with `reuse_existing_data_store`) to evaluate an existing data store as-is. |
| `reuse_existing_data_store` | `false` | When true, preparation validates the existing data store instead of creating or importing into one. |
| `location` | `global` | Vertex AI Search region: `global`, `us`, or `eu`. |
| `staging_bucket` | empty | Leave blank in the platform to use managed/configured GCS staging. |
| `num_previous_chunks` | `2` | Adjacent chunks returned before each matched chunk (0-3). |
| `num_next_chunks` | `2` | Adjacent chunks returned after each matched chunk (0-3). |
| `generation_mode` | `framework` | `framework` uses the standard LLM pipeline; `google_grounded` uses Vertex AI Search's grounded answer generation. |

`data_store_id`, `reuse_existing_data_store`, `location`, and `staging_bucket` are
build-time settings. `num_previous_chunks`, `num_next_chunks`, and `generation_mode`
are query-time settings that can be changed for a run against a ready index.

CLI environment:

- `GOOGLE_VERTEX_PROJECT_ID`
- `GOOGLE_VERTEX_LOCATION`
- `GOOGLE_VERTEX_SA_KEY_PATH` (optional; falls back to Application Default Credentials)
- `GOOGLE_VERTEX_DATA_STORE_ID`
- `GOOGLE_VERTEX_STAGING_BUCKET`
- `GOOGLE_VERTEX_GENERATION_MODE`

## Choosing A Strategy

Start with `vector_semantic` to establish a baseline. Add `vector_hybrid` if exact
terms matter. Try `graph_rag` when relationships are central to your questions. Use
`filesystem_rag` or `rlm_rag` when corpora are large enough that agentic exploration is
worth the latency and cost trade-off.

For reliable comparisons:

1. Use the same knowledge base and test set for each strategy.
2. Build a separate index for each RAG configuration.
3. Run the same selected metrics.
4. Compare quality, latency, token usage, and cost together.
