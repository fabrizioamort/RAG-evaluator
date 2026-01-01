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

    # Graph RAG Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="", description="Neo4j password")

    # Evaluation Configuration
    eval_test_set_path: str = Field(
        default="data/test_set.json", description="Path to evaluation test set"
    )

    # Data directories
    raw_data_dir: str = Field(default="data/raw", description="Raw documents directory")
    processed_data_dir: str = Field(
        default="data/processed", description="Processed documents directory"
    )


settings = Settings()
