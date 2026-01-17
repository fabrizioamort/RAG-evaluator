"""Tests for LLM provider service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm_provider import (
    LLMCompletionResponse,
    LLMEmbeddingResponse,
    LLMProviderService,
)


@pytest.fixture
def llm_service() -> LLMProviderService:
    """Create an LLM provider service instance."""
    return LLMProviderService()


class TestLLMProviderService:
    """Tests for LLMProviderService."""

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_completion_success(
        self, mock_acompletion: AsyncMock, llm_service: LLMProviderService
    ) -> None:
        """Test successful completion call."""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_response.get.side_return = lambda key, default=None: {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_total_cost": 0.001,
        }.get(key, default)

        # MagicMock's get is a bit tricky, let's use a dict-like mock if possible
        mock_response.__getitem__.side_effect = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_total_cost": 0.001,
        }.__getitem__
        mock_response.get.side_effect = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "_total_cost": 0.001,
        }.get

        mock_acompletion.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        response = await llm_service.completion(
            model="gpt-4o-mini", messages=messages, provider="openai"
        )

        assert isinstance(response, LLMCompletionResponse)
        assert response.content == "Test answer"
        assert response.usage.total_tokens == 15
        assert response.usage.cost_usd == 0.001
        assert response.model == "gpt-4o-mini"
        assert response.provider == "openai"
        assert response.latency_seconds >= 0

        mock_acompletion.assert_called_once()
        args, kwargs = mock_acompletion.call_args
        assert kwargs["model"] == "openai/gpt-4o-mini"

    @pytest.mark.asyncio
    @patch("litellm.aembedding")
    async def test_get_embedding_success(
        self, mock_aembedding: AsyncMock, llm_service: LLMProviderService
    ) -> None:
        """Test successful embedding call."""
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock_response.get.side_effect = {
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
            "_total_cost": 0.0001,
        }.get

        mock_aembedding.return_value = mock_response

        response = await llm_service.get_embedding(
            model="text-embedding-3-small", input_text="Test text", provider="openai"
        )

        assert isinstance(response, LLMEmbeddingResponse)
        assert response.embedding == [0.1, 0.2, 0.3]
        assert response.usage.prompt_tokens == 8
        assert response.model == "text-embedding-3-small"

        mock_aembedding.assert_called_once()
        args, kwargs = mock_aembedding.call_args
        assert kwargs["model"] == "openai/text-embedding-3-small"

    def test_count_tokens(self, llm_service: LLMProviderService) -> None:
        """Test token counting."""
        with patch("litellm.token_counter") as mock_counter:
            mock_counter.return_value = 10

            count = llm_service.count_tokens("gpt-4", "This is a test.")

            assert count == 10
            mock_counter.assert_called_once_with(model="gpt-4", text="This is a test.")

    def test_count_tokens_fallback(self, llm_service: LLMProviderService) -> None:
        """Test token counting fallback on error."""
        with patch("litellm.token_counter") as mock_counter:
            mock_counter.side_effect = Exception("Not supported")

            text = "This is a test."
            count = llm_service.count_tokens("unknown-model", text)

            # Simple fallback is len(text) // 4
            assert count == len(text) // 4

    @pytest.mark.asyncio
    @patch("litellm.acompletion")
    async def test_completion_with_base_url(
        self, mock_acompletion: AsyncMock, llm_service: LLMProviderService
    ) -> None:
        """Test completion with custom base URL (e.g. for Ollama)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Ollama response"
        mock_response.get.side_effect = {}.get  # No usage info for simple mocks

        mock_acompletion.return_value = mock_response

        response = await llm_service.completion(
            model="llama3",
            messages=[{"role": "user", "content": "Hi"}],
            provider="ollama",
            base_url="http://localhost:11434",
        )

        assert response.content == "Ollama response"
        mock_acompletion.assert_called_once()
        args, kwargs = mock_acompletion.call_args
        assert kwargs["api_base"] == "http://localhost:11434"
        assert kwargs["model"] == "ollama/llama3"
