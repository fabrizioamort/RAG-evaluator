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
