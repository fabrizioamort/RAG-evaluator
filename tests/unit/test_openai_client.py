"""Unit tests for the OpenAI-compatible client factory (incl. Vertex AI branch)."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest

from rag_evaluator.common.base_rag import RAGConfig
from rag_evaluator.common.openai_client import (
    embedding_client,
    embedding_openai_kwargs,
    is_vertex_ai_provider,
    llm_client,
    llm_openai_kwargs,
    make_client,
    resolve_embedding_model,
    resolve_llm_model,
)


@pytest.fixture(autouse=True)
def _reset_token_singleton():
    """Isolate the GCP token singleton between tests."""
    from rag_evaluator.common.gcp_token_provider import GCPTokenProvider

    GCPTokenProvider.reset()
    yield
    GCPTokenProvider.reset()


def _make_creds(token: str = "test-token") -> MagicMock:
    from datetime import datetime, timedelta

    creds = MagicMock()
    creds.token = token
    creds.expiry = (datetime.now(UTC) + timedelta(seconds=3600)).replace(tzinfo=None)
    return creds


def test_is_vertex_ai_provider_recognizes_aliases():
    assert is_vertex_ai_provider("vertex_ai")
    assert is_vertex_ai_provider("Vertex_AI")
    assert is_vertex_ai_provider("gemini")
    assert is_vertex_ai_provider("google_vertex_ai")
    assert not is_vertex_ai_provider("openai")
    assert not is_vertex_ai_provider("")
    assert not is_vertex_ai_provider(None)


def test_make_client_openai_default_passes_settings_defaults():
    with patch("rag_evaluator.common.openai_client.OpenAI") as mock_openai:
        make_client(api_key="key-x", base_url="https://api.example/v1", timeout=30)
        mock_openai.assert_called_once_with(
            api_key="key-x", base_url="https://api.example/v1", timeout=30
        )


def test_make_client_vertex_ai_uses_openai_compat_endpoint():
    creds = _make_creds("tok-abc")
    with (
        patch(
            "rag_evaluator.common.gcp_token_provider._google_auth_default",
            return_value=(creds, "auto-detected"),
        ),
        patch("rag_evaluator.common.openai_client.settings") as mock_settings,
        patch("rag_evaluator.common.openai_client.OpenAI") as mock_openai,
    ):
        mock_settings.google_cloud_project = "my-proj"
        mock_settings.google_cloud_location = "us-central1"
        mock_settings.google_vertex_project_id = ""
        mock_settings.openai_timeout = 120

        make_client(api_key="ignored", base_url="ignored", timeout=60, provider="vertex_ai")

        args, kwargs = mock_openai.call_args
        assert kwargs["base_url"] == (
            "https://us-central1-aiplatform.googleapis.com/v1beta1/"
            "projects/my-proj/locations/us-central1/endpoints/openapi"
        )
        # api_key is a non-empty placeholder; the httpx Auth hook injects the real token
        assert kwargs["api_key"]
        assert "http_client" in kwargs


def test_make_client_vertex_ai_falls_back_to_vertex_search_project():
    creds = _make_creds("tok")
    with (
        patch(
            "rag_evaluator.common.gcp_token_provider._google_auth_default",
            return_value=(creds, None),
        ),
        patch("rag_evaluator.common.openai_client.settings") as mock_settings,
        patch("rag_evaluator.common.openai_client.OpenAI") as mock_openai,
    ):
        mock_settings.google_cloud_project = None
        mock_settings.google_vertex_project_id = "fallback-proj"
        mock_settings.google_cloud_location = "europe-west4"
        mock_settings.openai_timeout = 30

        make_client(api_key=None, base_url=None, timeout=30, provider="vertex_ai")
        _, kwargs = mock_openai.call_args
        assert "projects/fallback-proj" in kwargs["base_url"]
        assert "europe-west4" in kwargs["base_url"]


def test_llm_client_routes_by_config_provider():
    config = RAGConfig(name="t", llm_provider="openai", llm_model="gpt-4o", llm_api_key="k")
    with patch("rag_evaluator.common.openai_client.make_client") as mock_mk:
        llm_client(config)
        mock_mk.assert_called_once()
        _, kwargs = mock_mk.call_args
        assert kwargs["provider"] == "openai"


def test_embedding_client_uses_embedding_provider_when_set():
    config = RAGConfig(
        name="t",
        llm_provider="openai",
        embedding_provider="vertex_ai",
        embedding_model="gemini-embedding-2",
    )
    with patch("rag_evaluator.common.openai_client.make_client") as mock_mk:
        embedding_client(config)
        _, kwargs = mock_mk.call_args
        assert kwargs["provider"] == "vertex_ai"


def test_embedding_client_falls_back_to_llm_provider():
    config = RAGConfig(name="t", llm_provider="vertex_ai", embedding_provider="")
    with patch("rag_evaluator.common.openai_client.make_client") as mock_mk:
        embedding_client(config)
        _, kwargs = mock_mk.call_args
        assert kwargs["provider"] == "vertex_ai"


def test_resolve_llm_model_prepends_google_prefix_for_vertex_ai():
    config = RAGConfig(name="t", llm_provider="vertex_ai", llm_model="gemini-2.5-pro")
    assert resolve_llm_model(config) == "google/gemini-2.5-pro"
    # explicit override
    assert resolve_llm_model(config, "gemini-2.5-flash") == "google/gemini-2.5-flash"


def test_resolve_llm_model_no_change_for_openai():
    config = RAGConfig(name="t", llm_provider="openai", llm_model="gpt-4o-mini")
    assert resolve_llm_model(config) == "gpt-4o-mini"


def test_resolve_llm_model_passthrough_when_already_prefixed():
    config = RAGConfig(name="t", llm_provider="vertex_ai", llm_model="google/gemini-2.5-pro")
    assert resolve_llm_model(config) == "google/gemini-2.5-pro"


def test_resolve_embedding_model_uses_embedding_provider():
    config = RAGConfig(
        name="t",
        llm_provider="openai",
        embedding_provider="vertex_ai",
        embedding_model="gemini-embedding-2",
    )
    assert resolve_embedding_model(config) == "google/gemini-embedding-2"


def test_llm_openai_kwargs_vertex_returns_token_and_base_url():
    creds = _make_creds("kw-token")
    with (
        patch(
            "rag_evaluator.common.gcp_token_provider._google_auth_default",
            return_value=(creds, None),
        ),
        patch("rag_evaluator.common.openai_client.settings") as mock_settings,
    ):
        mock_settings.google_cloud_project = "p1"
        mock_settings.google_vertex_project_id = ""
        mock_settings.google_cloud_location = "us-central1"
        config = RAGConfig(name="t", llm_provider="vertex_ai")
        kw = llm_openai_kwargs(config)
        assert kw["api_key"] == "kw-token"
        assert "projects/p1" in kw["base_url"]


def test_embedding_openai_kwargs_openai_default_untouched():
    config = RAGConfig(
        name="t",
        llm_provider="openai",
        embedding_api_key="ek",
        embedding_base_url="https://emb.example/v1",
    )
    kw = embedding_openai_kwargs(config)
    assert kw == {"api_key": "ek", "base_url": "https://emb.example/v1"}
