"""Tests for the Vertex AI entry in the LLM model catalog."""

from __future__ import annotations

from app.services.llm_model_catalog import (
    EMBEDDING_MODEL_CATALOG,
    MODEL_CATALOG,
    get_embedding_models,
    get_models,
)


def test_vertex_ai_generation_models_present():
    assert "vertex_ai" in MODEL_CATALOG
    names = get_models("vertex_ai")
    assert "gemini-2.5-pro" in names
    assert "gemini-2.5-flash" in names
    assert "gemini-2.5-flash-lite" in names


def test_vertex_ai_embedding_models_present():
    assert "vertex_ai" in EMBEDDING_MODEL_CATALOG
    names = get_embedding_models("vertex_ai")
    assert "gemini-embedding-2" in names


def test_openai_embedding_backfill_present():
    names = get_embedding_models("openai")
    assert "text-embedding-3-small" in names


def test_get_embedding_models_unknown_provider_returns_empty():
    assert get_embedding_models("unknown-provider") == []
