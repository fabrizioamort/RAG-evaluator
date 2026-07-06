"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=[".env", "../../.env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./storage/dev.db"

    # Storage
    STORAGE_PATH: str = "./storage"

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"

    # API
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # LLM Providers
    OPENAI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Default LLM settings
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
    LLM_COMPLETION_TIMEOUT_SECONDS: float = 120.0
    LLM_COMPLETION_RETRY_ATTEMPTS: int = 3
    LLM_COMPLETION_RETRY_BASE_DELAY_SECONDS: float = 1.0

    # Evaluation settings
    EVAL_CHECKPOINT_INTERVAL: int = 5  # Checkpoint every N test cases
    EVAL_MAX_CONCURRENT: int = 1  # Max concurrent evaluations (OSS: 1)
    EVAL_INCLUDE_REASON: bool = True  # Whether to include reasoning in metrics
    EVAL_G_EVAL_THRESHOLD: float = 0.7  # Default threshold for G-Eval

    # DeepEval Parallel Evaluation
    DEEPEVAL_ASYNC_MODE: bool = False
    DEEPEVAL_MAX_CONCURRENCY: int = 5

    # Neo4j settings (used for graph_rag connection validation)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # Webhook settings
    WEBHOOK_MAX_PER_PROJECT: int = 3
    WEBHOOK_TIMEOUT_SECONDS: int = 30
    WEBHOOK_MAX_RETRIES: int = 3

    # Version info
    APP_VERSION: str = "0.1.0"
    APP_NAME: str = "RAG Evaluation Platform"

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return "sqlite" in self.DATABASE_URL.lower()

    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL database."""
        return "postgresql" in self.DATABASE_URL.lower()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
