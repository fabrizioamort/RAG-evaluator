"""Unit tests for the DeepEval judge builder."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_token_singleton():
    from rag_evaluator.common.gcp_token_provider import GCPTokenProvider

    GCPTokenProvider.reset()
    yield
    GCPTokenProvider.reset()


def test_build_judge_model_defaults_to_openai_model_string():
    from rag_evaluator.evaluation import evaluator as ev

    with patch.object(ev, "settings") as mock_settings:
        mock_settings.judge_provider = None
        mock_settings.judge_model = None
        mock_settings.openai_model = "gpt-4o-mini"
        assert ev._build_judge_model() == "gpt-4o-mini"


def test_build_judge_model_openai_uses_judge_model_when_provided():
    from rag_evaluator.evaluation import evaluator as ev

    with patch.object(ev, "settings") as mock_settings:
        mock_settings.judge_provider = "openai"
        mock_settings.judge_model = "gpt-5"
        mock_settings.openai_model = "gpt-4o-mini"
        assert ev._build_judge_model() == "gpt-5"


def test_build_judge_model_vertex_ai_returns_custom_wrapper():
    from datetime import datetime, timedelta

    from rag_evaluator.evaluation import evaluator as ev

    creds = MagicMock()
    creds.token = "test-tok"
    creds.expiry = (datetime.now(UTC) + timedelta(seconds=3600)).replace(tzinfo=None)

    with (
        patch.object(ev, "settings") as mock_settings,
        patch(
            "rag_evaluator.common.gcp_token_provider._google_auth_default",
            return_value=(creds, "detected-proj"),
        ),
        patch("rag_evaluator.common.openai_client.settings") as mock_oc_settings,
    ):
        mock_settings.judge_provider = "vertex_ai"
        mock_settings.judge_model = "gemini-2.5-pro"
        mock_settings.openai_model = "gpt-4o-mini"
        mock_settings.vertex_gemini_model = "gemini-2.5-pro"
        mock_oc_settings.google_cloud_project = "p"
        mock_oc_settings.google_vertex_project_id = ""
        mock_oc_settings.google_cloud_location = "us-central1"
        mock_oc_settings.openai_timeout = 30

        judge = ev._build_judge_model()
        # must be a DeepEvalBaseLLM subclass (not a string)
        assert not isinstance(judge, str)
        assert "Vertex AI" in judge.get_model_name()
        assert "google/gemini-2.5-pro" in judge.get_model_name()
