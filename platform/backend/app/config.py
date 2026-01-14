"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
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
    ANTHROPIC_API_KEY: str | None = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Default LLM settings
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"

    # Evaluation settings
    EVAL_CHECKPOINT_INTERVAL: int = 5  # Checkpoint every N test cases
    EVAL_MAX_CONCURRENT: int = 1  # Max concurrent evaluations (OSS: 1)
    EVAL_INCLUDE_REASON: bool = True  # Whether to include reasoning in metrics

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
