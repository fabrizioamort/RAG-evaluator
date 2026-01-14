"""Configuration management for RAG Evaluator."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model to use")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )

    # Vector DB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", description="ChromaDB persistence directory"
    )

    # Qdrant Configuration
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant HTTP endpoint")
    qdrant_collection_name: str = Field(
        default="hybrid_rag", description="Qdrant collection name for hybrid search"
    )

    # Hybrid Search Configuration
    hybrid_chunk_size: int = Field(
        default=700, description="Chunk size for hybrid search (smaller for keyword matching)"
    )
    hybrid_chunk_overlap: int = Field(default=100, description="Chunk overlap for hybrid search")
    hybrid_fusion_alpha: float = Field(
        default=0.5, description="Weight for dense vs sparse (0=sparse only, 1=dense only)"
    )
    sparse_model_name: str = Field(
        default="prithivida/Splade_PP_en_v1",
        description="FastEmbed sparse model for keyword search",
    )

    # Graph RAG Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="", description="Neo4j password")

    # Evaluation Configuration
    eval_test_set_path: str = Field(
        default="data/test_set.json", description="Path to evaluation test set"
    )
    eval_reports_dir: str = Field(default="reports", description="Evaluation reports directory")

    # DeepEval Metric Thresholds (0.0 to 1.0)
    eval_faithfulness_threshold: float = Field(
        default=0.7, description="Minimum threshold for faithfulness metric"
    )
    eval_answer_relevancy_threshold: float = Field(
        default=0.7, description="Minimum threshold for answer relevancy metric"
    )
    eval_contextual_precision_threshold: float = Field(
        default=0.7, description="Minimum threshold for contextual precision metric"
    )
    eval_contextual_recall_threshold: float = Field(
        default=0.7, description="Minimum threshold for contextual recall metric"
    )
    eval_include_reason: bool = Field(
        default=True, description="Whether to include reasoning in DeepEval metrics"
    )

    # Data directories
    raw_data_dir: str = Field(default="data/raw", description="Raw documents directory")
    processed_data_dir: str = Field(
        default="data/processed", description="Processed documents directory"
    )

    # DeepEval Configuration
    deepeval_async_mode: bool = Field(
        default=False, description="Enable async mode for DeepEval (may hit rate limits)"
    )
    openai_timeout: int = Field(
        default=600, description="OpenAI API timeout in seconds (default: 600s/10min)"
    )
    deepeval_per_task_timeout: int = Field(
        default=900, description="DeepEval per-task timeout in seconds (default: 900s/15min)"
    )
    deepeval_per_attempt_timeout: int = Field(
        default=300, description="DeepEval per-attempt timeout in seconds (default: 300s/5min)"
    )
    deepeval_max_retries: int = Field(
        default=3, description="DeepEval maximum retry attempts (default: 3)"
    )
    deepeval_max_concurrent: int = Field(
        default=10, description="Maximum concurrent DeepEval tasks (default: 10)"
    )
    deepeval_throttle_value: float = Field(
        default=0.0, description="DeepEval throttle value in seconds (default: 0.0)"
    )


settings = Settings()
