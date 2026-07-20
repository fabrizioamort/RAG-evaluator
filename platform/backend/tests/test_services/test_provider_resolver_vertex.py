"""Tests for the Vertex AI branch of provider_resolver."""

from __future__ import annotations

from unittest.mock import patch

from app.services.provider_resolver import (
    ProviderEndpoint,
    is_vertex_ai_provider,
    resolve_provider_endpoint,
)


def test_is_vertex_ai_provider_recognizes_aliases():
    assert is_vertex_ai_provider("vertex_ai")
    assert is_vertex_ai_provider("Vertex_AI")
    assert is_vertex_ai_provider("gemini")
    assert is_vertex_ai_provider("google_vertex_ai")
    assert not is_vertex_ai_provider("openai")
    assert not is_vertex_ai_provider("")
    assert not is_vertex_ai_provider(None)


def test_resolve_provider_endpoint_openai_keeps_prior_shape():
    with patch("app.services.provider_resolver.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = "sk-x"
        ep = resolve_provider_endpoint("openai")
        assert ep.api_key == "sk-x"
        assert ep.base_url is None
        assert ep.vertex_project is None
        assert ep.vertex_location is None


def test_resolve_provider_endpoint_vertex_ai_uses_gcp_settings():
    with patch("app.services.provider_resolver.settings") as mock_settings:
        mock_settings.GOOGLE_CLOUD_PROJECT = "my-proj"
        mock_settings.GOOGLE_CLOUD_LOCATION = "us-central1"
        ep = resolve_provider_endpoint("vertex_ai")
        assert isinstance(ep, ProviderEndpoint)
        assert ep.api_key is None  # ADC handles auth
        assert ep.base_url is None
        assert ep.vertex_project == "my-proj"
        assert ep.vertex_location == "us-central1"


def test_resolve_provider_endpoint_vertex_ai_honors_base_url_override():
    with patch("app.services.provider_resolver.settings") as mock_settings:
        mock_settings.GOOGLE_CLOUD_PROJECT = "p"
        mock_settings.GOOGLE_CLOUD_LOCATION = "eu"
        ep = resolve_provider_endpoint("vertex_ai", base_url_override="https://custom")
        assert ep.base_url == "https://custom"
        assert ep.vertex_project == "p"
