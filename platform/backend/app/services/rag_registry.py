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
                        description="ChromaDB collection name",
                        default="rag_documents",
                    ),
                    RAGTypeParameter(
                        name="persist_directory",
                        type="string",
                        description="Custom persistence directory",
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
                        description="Qdrant collection name",
                        required=False,
                    ),
                    RAGTypeParameter(
                        name="qdrant_url",
                        type="string",
                        description="Qdrant server URL",
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
                        description="Neo4j connection URI",
                        required=False,
                    ),
                    RAGTypeParameter(
                        name="neo4j_username",
                        type="string",
                        description="Neo4j username",
                        required=False,
                    ),
                    RAGTypeParameter(
                        name="neo4j_password",
                        type="string",
                        description="Neo4j password",
                        required=False,
                    ),
                    RAGTypeParameter(
                        name="vector_index_name",
                        type="string",
                        description="Name of the vector index",
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
                        description="Model to use for agent navigation",
                        default="gpt-4o-mini",
                    ),
                    RAGTypeParameter(
                        name="prepared_path",
                        type="string",
                        description="Path for prepared filesystem output",
                        default="data/prepared/filesystem_rag",
                    ),
                    RAGTypeParameter(
                        name="word_threshold",
                        type="integer",
                        description="Threshold for LLM vs heuristic analysis",
                        default=1000,
                    ),
                    RAGTypeParameter(
                        name="max_iterations",
                        type="integer",
                        description="Max agent iterations per query",
                        default=10,
                    ),
                    RAGTypeParameter(
                        name="max_tool_calls",
                        type="integer",
                        description="Max tool calls per query",
                        default=20,
                    ),
                    RAGTypeParameter(
                        name="max_file_reads",
                        type="integer",
                        description="Max file reads per query",
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
