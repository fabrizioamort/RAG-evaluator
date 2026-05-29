"""RAG type and LLM provider registry."""

from app.schemas.rag_config import LLMProviderInfo, RAGTypeInfo, RAGTypeParameter


class RAGRegistry:
    """Registry for available RAG implementations and LLM providers."""

    @staticmethod
    def get_rag_types() -> list[RAGTypeInfo]:
        """Get all available RAG implementation types."""
        return [
            RAGTypeInfo(
                name="vector_semantic",
                display_name="Vector Semantic Search",
                description="Standard dense vector retrieval using ChromaDB.",
                requires_index=True,
                parameters=[
                    RAGTypeParameter(
                        name="collection_name",
                        type="string",
                        description=(
                            "ChromaDB collection name. The platform creates a per-index "
                            "collection by default; set this only to reuse an existing collection."
                        ),
                        default="rag_documents",
                    ),
                    RAGTypeParameter(
                        name="persist_directory",
                        type="string",
                        description=(
                            "Filesystem path for Chroma persistence. Leave blank to use "
                            "platform-managed storage or the core default."
                        ),
                        required=False,
                    ),
                ],
            ),
            RAGTypeInfo(
                name="vector_hybrid",
                display_name="Hybrid Search",
                description="Combines dense and sparse vectors using Qdrant and RRF.",
                requires_index=True,
                parameters=[
                    RAGTypeParameter(
                        name="collection_name",
                        type="string",
                        description=(
                            "Qdrant collection name. The platform uses a per-index collection "
                            "by default; leave blank unless reusing an existing collection."
                        ),
                        required=False,
                    ),
                    RAGTypeParameter(
                        name="qdrant_url",
                        type="string",
                        description=(
                            "Qdrant server URL. Leave blank to use QDRANT_URL from .env or the "
                            "core default."
                        ),
                        required=False,
                    ),
                ],
            ),
            RAGTypeInfo(
                name="graph_rag",
                display_name="Graph RAG",
                description="Hybrid vector + graph traversal retrieval using Neo4j.",
                requires_index=True,
                parameters=[
                    RAGTypeParameter(
                        name="neo4j_uri",
                        type="string",
                        description=(
                            "Neo4j connection URI. Leave blank to use NEO4J_URI from .env "
                            "or the core default."
                        ),
                        required=False,
                    ),
                    RAGTypeParameter(
                        name="neo4j_username",
                        type="string",
                        description=(
                            "Neo4j username. Leave blank to use NEO4J_USERNAME from .env "
                            "or the core default."
                        ),
                        required=False,
                    ),
                    RAGTypeParameter(
                        name="neo4j_password",
                        type="string",
                        description=(
                            "Neo4j password. Leave blank to use NEO4J_PASSWORD from .env "
                            "or the core default."
                        ),
                        required=False,
                    ),
                    RAGTypeParameter(
                        name="vector_index_name",
                        type="string",
                        description=(
                            "Name of the Neo4j vector index. Keep the default unless you "
                            "created a custom index."
                        ),
                        default="chunk_embeddings",
                    ),
                ],
            ),
            RAGTypeInfo(
                name="filesystem_rag",
                display_name="Filesystem RAG",
                description="LLM-guided agent navigating a prepared filesystem.",
                requires_index=True,
                parameters=[
                    RAGTypeParameter(
                        name="llm_model",
                        type="string",
                        description=(
                            "Model used for agent navigation. In the platform, this is driven "
                            "by the LLM Settings section; the default is usually fine."
                        ),
                        default="gpt-4o-mini",
                    ),
                    RAGTypeParameter(
                        name="prepared_path",
                        type="string",
                        description=(
                            "Path for prepared filesystem output. The platform stores this under "
                            "storage/indexes/<index_id>/filesystem_rag by default."
                        ),
                        default="data/prepared/filesystem_rag",
                    ),
                    RAGTypeParameter(
                        name="word_threshold",
                        type="integer",
                        description=(
                            "Word count threshold for LLM vs heuristic analysis. Lower values "
                            "use the LLM more (higher cost)."
                        ),
                        default=1000,
                    ),
                    RAGTypeParameter(
                        name="max_iterations",
                        type="integer",
                        description="Max agent iterations per query.",
                        default=10,
                    ),
                    RAGTypeParameter(
                        name="max_tool_calls",
                        type="integer",
                        description="Max tool calls per query.",
                        default=20,
                    ),
                    RAGTypeParameter(
                        name="max_file_reads",
                        type="integer",
                        description="Max file reads per query.",
                        default=10,
                    ),
                ],
            ),
            RAGTypeInfo(
                name="rlm_rag",
                display_name="RLM-RAG",
                description=(
                    "Recursive language-model RAG that explores a prepared filesystem "
                    "with Python tools."
                ),
                requires_index=True,
                parameters=[
                    RAGTypeParameter(
                        name="security_mode",
                        type="string",
                        description=(
                            "Execution security mode. Use lite for trusted corpora; use full "
                            "for subprocess isolation with stricter path controls."
                        ),
                        default="lite",
                        choices=["lite", "full"],
                    ),
                    RAGTypeParameter(
                        name="orchestrator_model",
                        type="string",
                        description=(
                            "Model used for main reasoning and code generation. Leave blank "
                            "to use the LLM Settings model."
                        ),
                        required=False,
                    ),
                    RAGTypeParameter(
                        name="worker_model",
                        type="string",
                        description="Model used for summaries, topics, and sub-LLM calls.",
                        default="gpt-5-nano",
                    ),
                    RAGTypeParameter(
                        name="max_repl_steps",
                        type="integer",
                        description="Maximum Python exploration steps per query.",
                        default=15,
                        min_value=1,
                        max_value=50,
                    ),
                    RAGTypeParameter(
                        name="repl_timeout",
                        type="float",
                        description="Timeout in seconds for each REPL step.",
                        default=5.0,
                        min_value=0.1,
                        max_value=60,
                    ),
                    RAGTypeParameter(
                        name="max_file_reads",
                        type="integer",
                        description="Maximum file reads per query.",
                        default=12,
                    ),
                    RAGTypeParameter(
                        name="max_read_bytes",
                        type="integer",
                        description="Maximum bytes returned by a file read.",
                        default=50000,
                    ),
                    RAGTypeParameter(
                        name="max_read_lines",
                        type="integer",
                        description="Maximum lines returned by a file read.",
                        default=1000,
                    ),
                    RAGTypeParameter(
                        name="max_sub_calls",
                        type="integer",
                        description="Maximum recursive worker-model calls per query.",
                        default=8,
                    ),
                    RAGTypeParameter(
                        name="max_recursion_depth",
                        type="integer",
                        description="Maximum nested sub-LLM call depth.",
                        default=2,
                    ),
                    RAGTypeParameter(
                        name="small_corpus_threshold",
                        type="integer",
                        description=(
                            "Use the simple-context fallback at or below this document count."
                        ),
                        default=10,
                    ),
                    RAGTypeParameter(
                        name="chunk_size",
                        type="integer",
                        description="Preparation chunk size.",
                        default=1000,
                    ),
                    RAGTypeParameter(
                        name="chunk_overlap",
                        type="integer",
                        description="Preparation chunk overlap.",
                        default=200,
                    ),
                    RAGTypeParameter(
                        name="use_llm_summaries",
                        type="boolean",
                        description="Generate LLM summaries during preparation.",
                        default=True,
                    ),
                    RAGTypeParameter(
                        name="use_llm_topics",
                        type="boolean",
                        description="Extract LLM topics during preparation.",
                        default=True,
                    ),
                    RAGTypeParameter(
                        name="max_topics_per_doc",
                        type="integer",
                        description="Maximum topics extracted per document.",
                        default=5,
                    ),
                ],
            ),
        ]

    @staticmethod
    def get_llm_providers() -> list[LLMProviderInfo]:
        """Get all supported LLM providers."""
        return [
            LLMProviderInfo(
                name="openai",
                display_name="OpenAI",
                models=["gpt-5.1", "gpt-5-mini", "gpt-5-nano"],
                requires_api_key=True,
                supports_base_url=True,
            ),
            LLMProviderInfo(
                name="anthropic",
                display_name="Anthropic",
                models=["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"],
                requires_api_key=True,
                supports_base_url=False,
            ),
            LLMProviderInfo(
                name="ollama",
                display_name="Ollama (Local)",
                models=["llama3", "mistral", "phi3"],
                requires_api_key=False,
                supports_base_url=True,
            ),
        ]
