"""Tests for configuration management."""

from rag_evaluator.config import Settings


def test_settings_default_values() -> None:
    """Test that settings have correct default values."""
    settings = Settings()

    assert settings.openai_model == "gpt-4-turbo-preview"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_username == "neo4j"


def test_settings_data_directories() -> None:
    """Test that data directory settings are correct."""
    settings = Settings()

    assert settings.raw_data_dir == "data/raw"
    assert settings.processed_data_dir == "data/processed"
    assert settings.chroma_persist_directory == "./data/chroma_db"
