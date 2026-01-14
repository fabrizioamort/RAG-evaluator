"""Tests for the test quality gate service."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.services.test_quality_gate import (
    QualityConfig,
    TestQualityGateService,
    get_quality_gate_service,
)


@pytest.fixture
def quality_gate() -> TestQualityGateService:
    """Create a quality gate service for testing."""
    config = QualityConfig(
        semantic_duplicate_threshold=0.92,
        min_question_words=4,
        max_question_words=100,
        min_answer_words=2,
        answerability_threshold=0.25,
    )
    return TestQualityGateService(config=config)


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Create a mock LLM service."""
    mock = MagicMock()
    mock.get_embedding = AsyncMock()
    return mock


class TestQualityGateBasicChecks:
    """Tests for basic quality gate checks."""

    @pytest.mark.asyncio
    async def test_question_too_short(self, quality_gate: TestQualityGateService) -> None:
        """Test rejection of too short questions."""
        result = await quality_gate.validate(
            question="What?",
            expected_answer="This is a valid answer.",
            context=["Some context about the topic."],
            skip_semantic_check=True,
        )

        assert not result.passed
        assert result.rejection_reason is not None
        assert "too short" in result.rejection_reason.lower()
        assert result.details["length_check"]["passed"] is False

    @pytest.mark.asyncio
    async def test_question_too_long(self, quality_gate: TestQualityGateService) -> None:
        """Test rejection of too long questions."""
        long_question = " ".join(["word"] * 110)

        result = await quality_gate.validate(
            question=long_question,
            expected_answer="This is a valid answer.",
            context=["Some context about word and topic."],
            skip_semantic_check=True,
        )

        assert not result.passed
        assert result.rejection_reason is not None
        assert "too long" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_answer_too_short(self, quality_gate: TestQualityGateService) -> None:
        """Test rejection of too short answers."""
        result = await quality_gate.validate(
            question="What is the main concept being discussed here?",
            expected_answer="Yes",  # Only 1 word
            context=["Some context about the topic."],
            skip_semantic_check=True,
        )

        assert not result.passed
        assert result.rejection_reason is not None
        assert "answer" in result.rejection_reason.lower()
        assert "short" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_valid_question_passes(self, quality_gate: TestQualityGateService) -> None:
        """Test that a valid question passes all checks."""
        result = await quality_gate.validate(
            question="What is the main concept being discussed in the document?",
            expected_answer="The main concept is machine learning and its applications.",
            context=[
                "Machine learning is a field of artificial intelligence. "
                "It has many applications in various domains."
            ],
            skip_semantic_check=True,
        )

        assert result.passed
        assert result.rejection_reason is None
        assert result.score > 0.0


class TestExactDuplicateDetection:
    """Tests for exact duplicate detection."""

    @pytest.mark.asyncio
    async def test_exact_duplicate_rejected(self, quality_gate: TestQualityGateService) -> None:
        """Test that exact duplicate questions are rejected."""
        question = "What is the purpose of this document?"
        context = ["This document explains the purpose and scope."]
        answer = "The purpose is to explain the scope."

        # First question should pass
        result1 = await quality_gate.validate(
            question=question,
            expected_answer=answer,
            context=context,
            skip_semantic_check=True,
        )
        assert result1.passed

        # Exact duplicate should be rejected
        result2 = await quality_gate.validate(
            question=question,
            expected_answer=answer,
            context=context,
            skip_semantic_check=True,
        )
        assert not result2.passed
        assert result2.rejection_reason is not None
        assert "duplicate" in result2.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_normalized_duplicate_rejected(
        self, quality_gate: TestQualityGateService
    ) -> None:
        """Test that normalized duplicates are rejected."""
        context = ["This document explains the purpose and scope."]
        answer = "The purpose is to explain the scope."

        # First question
        await quality_gate.validate(
            question="What is the purpose of this document?",
            expected_answer=answer,
            context=context,
            skip_semantic_check=True,
        )

        # Same question with different case and punctuation
        result = await quality_gate.validate(
            question="WHAT IS THE PURPOSE OF THIS DOCUMENT",
            expected_answer=answer,
            context=context,
            skip_semantic_check=True,
        )
        assert not result.passed
        assert result.rejection_reason is not None
        assert "duplicate" in result.rejection_reason.lower()


class TestAnswerabilityValidation:
    """Tests for answerability validation."""

    @pytest.mark.asyncio
    async def test_answerable_question_passes(self, quality_gate: TestQualityGateService) -> None:
        """Test that an answerable question passes."""
        result = await quality_gate.validate(
            question="What is Python used for in this context?",
            expected_answer="Python is used for data analysis and machine learning.",
            context=[
                "Python is a programming language. It is used for data analysis. "
                "Python is also popular for machine learning tasks."
            ],
            skip_semantic_check=True,
        )

        assert result.passed
        assert result.details["answerability_check"]["passed"]
        assert result.details["answerability_check"]["overlap_score"] > 0.25

    @pytest.mark.asyncio
    async def test_unanswerable_question_rejected(
        self, quality_gate: TestQualityGateService
    ) -> None:
        """Test that an unanswerable question is rejected."""
        result = await quality_gate.validate(
            question="What is the capital of France according to this text?",
            expected_answer="Paris is the capital of France with a population of 2 million.",
            context=["Python is a programming language. It is used for web development."],
            skip_semantic_check=True,
        )

        assert not result.passed
        assert result.rejection_reason is not None
        assert (
            "derivable" in result.rejection_reason.lower()
            or "overlap" in result.rejection_reason.lower()
        )

    @pytest.mark.asyncio
    async def test_no_context_rejected(self, quality_gate: TestQualityGateService) -> None:
        """Test that questions with no context are rejected."""
        result = await quality_gate.validate(
            question="What is the main topic of this document?",
            expected_answer="The main topic is technology.",
            context=[],
            skip_semantic_check=True,
        )

        assert not result.passed
        assert result.rejection_reason is not None
        assert "context" in result.rejection_reason.lower()


class TestSemanticDuplicateDetection:
    """Tests for semantic duplicate detection."""

    @pytest.mark.asyncio
    async def test_semantic_duplicate_with_mock(self, mock_llm_service: MagicMock) -> None:
        """Test semantic duplicate detection with mocked embeddings."""
        # Create service with mock
        quality_gate = TestQualityGateService(llm_service=mock_llm_service)

        # Create similar embeddings
        embedding1 = np.random.rand(384).tolist()
        embedding2 = embedding1.copy()  # Same embedding = similarity 1.0

        # Mock embedding responses
        mock_response = MagicMock()
        mock_response.embedding = embedding1
        mock_llm_service.get_embedding.return_value = mock_response

        # First question passes - use context that contains answer terms
        context = ["The main topic discussed here is technology and computing."]
        result1 = await quality_gate.validate(
            question="What is the main topic discussed here?",
            expected_answer="The main topic is technology.",
            context=context,
        )
        assert result1.passed

        # Reset mock for second call - return very similar embedding
        mock_response.embedding = embedding2
        mock_llm_service.get_embedding.return_value = mock_response

        # Semantically similar question should be rejected
        result2 = await quality_gate.validate(
            question="What is the primary topic being discussed?",
            expected_answer="The primary topic is technology.",
            context=context,
        )
        assert not result2.passed
        assert result2.rejection_reason is not None
        assert "semantic" in result2.rejection_reason.lower()


class TestQualityScore:
    """Tests for quality score calculation."""

    @pytest.mark.asyncio
    async def test_high_quality_score(self, quality_gate: TestQualityGateService) -> None:
        """Test that high-quality questions get high scores."""
        result = await quality_gate.validate(
            question="What are the main advantages of using Python for data science?",
            expected_answer="The main advantages include extensive libraries, ease of use, and strong community support.",
            context=[
                "Python is popular for data science. Its main advantages include "
                "extensive libraries like pandas and numpy. Python is known for ease of use. "
                "The language has strong community support."
            ],
            skip_semantic_check=True,
        )

        assert result.passed
        assert result.score >= 0.7

    @pytest.mark.asyncio
    async def test_moderate_quality_score(self, quality_gate: TestQualityGateService) -> None:
        """Test quality score for moderately good questions."""
        result = await quality_gate.validate(
            question="What is Python used for?",
            expected_answer="Python is used for programming and data analysis.",
            context=["Python is a programming language. It is used for data analysis."],
            skip_semantic_check=True,
        )

        assert result.passed
        # Score should be reasonable (between 0.4 and 1.0)
        assert 0.4 <= result.score <= 1.0


class TestStateManagement:
    """Tests for quality gate state management."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, quality_gate: TestQualityGateService) -> None:
        """Test that reset clears tracking state."""
        context = ["Some context about the topic."]
        answer = "The answer about the topic."

        # Add a question
        await quality_gate.validate(
            question="What is the topic being discussed here?",
            expected_answer=answer,
            context=context,
            skip_semantic_check=True,
        )

        # Check state
        stats = quality_gate.get_stats()
        assert stats["tracked_questions"] == 1

        # Reset
        quality_gate.reset()

        # Check state is cleared
        stats = quality_gate.get_stats()
        assert stats["tracked_questions"] == 0
        assert stats["tracked_embeddings"] == 0

    @pytest.mark.asyncio
    async def test_add_existing_question(self, quality_gate: TestQualityGateService) -> None:
        """Test adding existing questions to prevent duplicates."""
        existing_question = "What is the main concept?"

        # Add as existing
        quality_gate.add_existing_question(existing_question)

        # Try to add same question through validation
        result = await quality_gate.validate(
            question=existing_question,
            expected_answer="The main concept is testing.",
            context=["Some context about the main concept and testing."],
            skip_semantic_check=True,
        )

        assert not result.passed
        assert result.rejection_reason is not None
        assert "duplicate" in result.rejection_reason.lower()

    def test_get_stats(self, quality_gate: TestQualityGateService) -> None:
        """Test getting statistics."""
        stats = quality_gate.get_stats()

        assert "tracked_questions" in stats
        assert "tracked_embeddings" in stats
        assert "config" in stats
        assert "semantic_threshold" in stats["config"]


class TestFactoryFunction:
    """Tests for the factory function."""

    def test_get_quality_gate_service_creates_new_instance(self) -> None:
        """Test that factory creates new instances."""
        service1 = get_quality_gate_service()
        service2 = get_quality_gate_service()

        # Should be different instances (not singleton)
        assert service1 is not service2

    def test_get_quality_gate_service_with_config(self) -> None:
        """Test factory with custom config."""
        config = QualityConfig(min_question_words=10)
        service = get_quality_gate_service(config=config)

        assert service.config.min_question_words == 10
