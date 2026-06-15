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
| `filesystem_rag` | Local prepared files | ReAct-style agent with file tools | Good for large corpora and research-style queries | Agent behavior is slower and less deterministic than vector search |
| `rlm_rag` | Local prepared files | Recursive language-model agent with Python tools | Strong for large corpora that need programmatic exploration | Executes generated Python; choose security mode carefully |
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
| `sparse_model_name` | `prithvida/Splade_pp_en_v1` | Sparse model used while building sparse vectors. |

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
- `SPARSE_MODEL_NAME`

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

Filesystem RAG converts documents into a navigable directory structure with summaries,
indexes, and line-numbered source files. An agent uses file tools such as listing,
reading, searching, and finding files to gather context before answering.

Best for:

- Larger corpora where loading many chunks into a prompt is inefficient.
- Broad research questions.
- Cases where source navigation and summaries are useful for debugging.

Prepared structure:

```text
prepared_path/
  _meta/
  _index/
  _summaries/
  documents/
```

Platform parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `llm_model` | `gpt-4o-mini` | Usually controlled by the RAG configuration LLM model. |
| `prepared_path` | `data/prepared/filesystem_rag` | The platform uses managed storage under `storage/indexes`. |
| `word_threshold` | `1000` | Lower values use the LLM more during preparation. |
| `max_iterations` | `10` | Maximum agent reasoning loops. |
| `max_tool_calls` | `20` | Maximum tool calls per query. |
| `max_file_reads` | `10` | Maximum file reads per query. |

`prepared_path` and `word_threshold` are build-time settings. `llm_model`,
`max_iterations`, `max_tool_calls`, and `max_file_reads` are query-time settings that
can be changed for a run against a ready index.

## RLM-RAG

Type key: `rlm_rag`

RLM-RAG is a recursive language-model approach for large corpora. It prepares documents
into a filesystem and lets an LLM orchestrator write Python exploration code. The agent
can call a smaller worker model for summaries, topic extraction, and recursive document
analysis.

Best for:

- Large corpora where static top-k retrieval is too shallow.
- Questions that benefit from programmatic filtering, grouping, or iteration.
- Experiments with recursive language-model retrieval.

Security modes:

| Mode | Use case | Behavior |
| --- | --- | --- |
| `lite` | Trusted local corpora | Faster in-process execution. |
| `full` | Less trusted document content | Subprocess isolation, stricter path controls, and prompt-injection wrapping. |

Platform parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `security_mode` | `lite` | Use `full` for stricter isolation. |
| `orchestrator_model` | RAG config model | Main reasoning and code-generation model. |
| `worker_model` | `gpt-5-nano` | Worker model for summaries and sub-calls. |
| `max_repl_steps` | `15` | Maximum Python exploration steps. |
| `repl_timeout` | `5.0` | Timeout per Python step in seconds. |
| `max_file_reads` | `12` | Maximum file reads per query. |
| `max_read_bytes` | `50000` | Maximum bytes returned by a file read. |
| `max_read_lines` | `1000` | Maximum lines returned by a file read. |
| `max_sub_calls` | `8` | Maximum recursive worker calls. |
| `max_recursion_depth` | `2` | Maximum nested worker-call depth. |
| `small_corpus_threshold` | `10` | Uses a simple-context fallback at or below this document count. |
| `chunk_size` | `1000` | Preparation chunk size. |
| `chunk_overlap` | `200` | Preparation chunk overlap. |
| `use_llm_summaries` | `true` | Generate summaries during preparation. |
| `use_llm_topics` | `true` | Extract topics during preparation. |
| `max_topics_per_doc` | `5` | Maximum topics per source document. |

RLM-RAG can be powerful, but it is intentionally more complex than the baseline
retrievers. Use the playground to inspect outputs before committing to large evaluation
runs.

RLM-RAG build-time parameters are the preparation controls: `chunk_size`,
`chunk_overlap`, `use_llm_summaries`, `use_llm_topics`, `max_topics_per_doc`, and
`worker_model`. Query-time parameters include `llm_model`, `orchestrator_model`,
`security_mode`, execution limits, read limits, recursion limits, and
`small_corpus_threshold`. If you override `llm_model` and do not explicitly set
`orchestrator_model`, the platform uses the overridden generation model as the effective
orchestrator model.

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
