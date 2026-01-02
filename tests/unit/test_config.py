"""Tests for configuration management."""

from unittest.mock import patch

from rag_evaluator.config import Settings


def test_settings_default_values() -> None:
    """Test that settings have correct default values when no .env file is present."""
    # Test with no environment file to verify actual defaults
    with patch.dict("os.environ", {}, clear=True):
        settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert settings.openai_model == "gpt-4-turbo-preview"
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.neo4j_uri == "bolt://localhost:7687"
        assert settings.neo4j_username == "neo4j"
        assert settings.deepeval_async_mode is False
        assert settings.openai_timeout == 600
        assert settings.deepeval_per_task_timeout == 900
        assert settings.deepeval_per_attempt_timeout == 300
        assert settings.deepeval_max_retries == 3


def test_settings_data_directories() -> None:
    """Test that data directory settings are correct."""
    settings = Settings()

    assert settings.raw_data_dir == "data/raw"
    assert settings.processed_data_dir == "data/processed"
    assert settings.chroma_persist_directory == "./data/chroma_db"
