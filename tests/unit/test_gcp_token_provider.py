"""Unit tests for the GCP access token provider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import httpx
import pytest

from rag_evaluator.common.gcp_token_provider import (
    GCPBearerAuth,
    GCPTokenProvider,
    build_vertex_openai_base_url,
    prepend_google_prefix,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test gets a fresh provider."""
    GCPTokenProvider.reset()
    yield
    GCPTokenProvider.reset()


def _make_creds(token: str = "tok-init", expires_in_seconds: int | None = 3600) -> MagicMock:
    creds = MagicMock()
    creds.token = token
    if expires_in_seconds is None:
        creds.expiry = None
    else:
        # google-auth stores expiry as a naive UTC datetime
        creds.expiry = (datetime.now(UTC) + timedelta(seconds=expires_in_seconds)).replace(
            tzinfo=None
        )
    return creds


def _refresh_side_effect(new_token: str, new_expires_in_seconds: int = 3600):
    def _side(request):  # noqa: ARG001 - google-auth passes a Request instance
        # simulate refresh by mutating the mock creds captured in closure
        creds.token = new_token
        creds.expiry = (
            datetime.now(UTC) + timedelta(seconds=new_expires_in_seconds)
        ).replace(tzinfo=None)

    creds = MagicMock()
    return _side, creds


def test_get_token_returns_current_token_when_valid():
    creds = _make_creds(token="valid-token", expires_in_seconds=3600)
    with patch(
        "rag_evaluator.common.gcp_token_provider._google_auth_default",
        return_value=(creds, "test-project"),
    ):
        provider = GCPTokenProvider()
        assert provider.get_token() == "valid-token"
    creds.refresh.assert_not_called()


def test_get_token_refreshes_when_token_missing():
    creds = _make_creds(token="", expires_in_seconds=3600)

    def _refresh(request):  # noqa: ARG001
        creds.token = "refreshed-token"
        creds.expiry = (datetime.now(UTC) + timedelta(seconds=3600)).replace(tzinfo=None)

    creds.refresh.side_effect = _refresh
    with patch(
        "rag_evaluator.common.gcp_token_provider._google_auth_default",
        return_value=(creds, "test-project"),
    ):
        provider = GCPTokenProvider()
        assert provider.get_token() == "refreshed-token"
    creds.refresh.assert_called_once()


def test_get_token_refreshes_when_within_threshold():
    # 60s remaining < 300s threshold → refresh
    creds = _make_creds(token="stale-token", expires_in_seconds=60)

    def _refresh(request):  # noqa: ARG001
        creds.token = "new-token"
        creds.expiry = (datetime.now(UTC) + timedelta(seconds=3600)).replace(tzinfo=None)

    creds.refresh.side_effect = _refresh
    with patch(
        "rag_evaluator.common.gcp_token_provider._google_auth_default",
        return_value=(creds, "test-project"),
    ):
        provider = GCPTokenProvider()
        assert provider.get_token() == "new-token"
    creds.refresh.assert_called_once()


def test_get_token_does_not_refresh_when_far_from_expiry():
    creds = _make_creds(token="fresh-token", expires_in_seconds=3000)
    with patch(
        "rag_evaluator.common.gcp_token_provider._google_auth_default",
        return_value=(creds, "test-project"),
    ):
        provider = GCPTokenProvider()
        for _ in range(5):
            assert provider.get_token() == "fresh-token"
    creds.refresh.assert_not_called()


def test_get_token_refreshes_when_expiry_none():
    creds = _make_creds(token="tok", expires_in_seconds=None)

    def _refresh(request):  # noqa: ARG001
        creds.token = "refreshed"
        creds.expiry = (datetime.now(UTC) + timedelta(seconds=3600)).replace(tzinfo=None)

    creds.refresh.side_effect = _refresh
    with patch(
        "rag_evaluator.common.gcp_token_provider._google_auth_default",
        return_value=(creds, "test-project"),
    ):
        provider = GCPTokenProvider()
        assert provider.get_token() == "refreshed"


def test_get_token_raises_on_empty_after_refresh():
    creds = _make_creds(token="", expires_in_seconds=60)
    creds.refresh.return_value = None  # simulates failed refresh not raising
    with patch(
        "rag_evaluator.common.gcp_token_provider._google_auth_default",
        return_value=(creds, "test-project"),
    ):
        provider = GCPTokenProvider()
        with pytest.raises(RuntimeError, match="empty"):
            provider.get_token()


def test_instance_is_singleton_across_calls():
    creds = _make_creds(token="t", expires_in_seconds=3600)
    with patch(
        "rag_evaluator.common.gcp_token_provider._google_auth_default",
        return_value=(creds, "test-project"),
    ) as mock_default:
        first = GCPTokenProvider.instance()
        second = GCPTokenProvider.instance()
        assert first is second
    mock_default.assert_called_once()


def test_bearer_auth_sets_authorization_header():
    creds = _make_creds(token="hdr-token", expires_in_seconds=3600)
    with patch(
        "rag_evaluator.common.gcp_token_provider._google_auth_default",
        return_value=(creds, "test-project"),
    ):
        provider = GCPTokenProvider()
        auth = GCPBearerAuth(provider=provider)
        request = httpx.Request("POST", "https://example.com/x")
        flow = auth.auth_flow(request)
        prepared = next(flow)
        assert prepared.headers["Authorization"] == "Bearer hdr-token"


def test_build_vertex_openai_base_url_shape():
    url = build_vertex_openai_base_url("my-proj", "us-central1")
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/"
        "projects/my-proj/locations/us-central1/endpoints/openapi"
    )


def test_build_vertex_openai_base_url_requires_project_and_location():
    with pytest.raises(ValueError, match="PROJECT"):
        build_vertex_openai_base_url("", "us-central1")
    with pytest.raises(ValueError, match="LOCATION"):
        build_vertex_openai_base_url("my-proj", "")


def test_prepend_google_prefix_only_when_missing():
    assert prepend_google_prefix("gemini-2.5-flash") == "google/gemini-2.5-flash"
    assert prepend_google_prefix("google/gemini-2.5-pro") == "google/gemini-2.5-pro"
    assert prepend_google_prefix("publishers/meta/models/llama-3") == (
        "publishers/meta/models/llama-3"
    )
    assert prepend_google_prefix("") == ""
