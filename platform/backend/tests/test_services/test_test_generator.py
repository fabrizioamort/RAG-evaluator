"""Tests for the test generator service."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm_provider import LLMCompletionResponse, TokenUsage
from app.services.test_generator_service import (
    GenerationConfig,
    TestGeneratorService,
    get_test_generator_service,
)
from app.services.test_quality_gate import QualityResult, TestQualityGateService


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Create a mock LLM service."""
    mock = MagicMock()
    mock.completion = AsyncMock()
    mock.get_embedding = AsyncMock()
    return mock


@pytest.fixture
def mock_artifact_store() -> MagicMock:
    """Create a mock artifact store."""
    mock = MagicMock()
    mock.store_json = AsyncMock()
    mock.KIND_PROVENANCE = "provenance"
    return mock


@pytest.fixture
def mock_quality_gate() -> MagicMock:
    """Create a mock quality gate service."""
    mock = MagicMock(spec=TestQualityGateService)
    mock.validate = AsyncMock()
    mock.add_existing_questions = AsyncMock()
    mock.reset = MagicMock()
    return mock


@pytest.fixture
def generation_config() -> GenerationConfig:
    """Create a test generation config."""
    return GenerationConfig(
        target_count=5,
        questions_per_chunk=2,
        chunk_size=1000,
        chunk_overlap=100,
        llm_model="gpt-4o-mini",
        llm_provider="openai",
        temperature=0.7,
        skip_semantic_check=True,
    )


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_default_difficulty_distribution(self) -> None:
        """Test default difficulty distribution."""
        config = GenerationConfig()

        assert config.difficulty_distribution is not None
        assert config.difficulty_distribution["easy"] == 0.3
        assert config.difficulty_distribution["medium"] == 0.5
        assert config.difficulty_distribution["hard"] == 0.2

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = GenerationConfig(
            target_count=50,
            questions_per_chunk=5,
            chunk_size=3000,
        )

        assert config.target_count == 50
        assert config.questions_per_chunk == 5
        assert config.chunk_size == 3000


class TestDocumentChunking:
    """Tests for document chunking."""

    @pytest.mark.asyncio
    async def test_chunk_documents(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test document chunking logic."""
        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
        )

        config = GenerationConfig(chunk_size=100, chunk_overlap=20)

        # Create a test document
        content = "This is a test document. " * 20  # ~500 characters
        documents = [("test.txt", content)]

        chunks = service._chunk_documents(documents, config)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should have metadata
        for chunk in chunks:
            assert "text" in chunk
            assert "source_file" in chunk
            assert "chunk_index" in chunk
            assert chunk["source_file"] == "test.txt"

    @pytest.mark.asyncio
    async def test_chunk_empty_documents(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test chunking empty documents."""
        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
        )

        config = GenerationConfig()
        chunks = service._chunk_documents([], config)

        assert chunks == []


class TestLLMResponseParsing:
    """Tests for LLM response parsing."""

    @pytest.mark.asyncio
    async def test_parse_valid_json_array(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test parsing valid JSON array response."""
        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
        )

        response = json.dumps(
            [
                {
                    "question": "What is Python?",
                    "expected_answer": "Python is a programming language.",
                    "difficulty": "easy",
                    "question_type": "factual",
                }
            ]
        )

        parsed = service._parse_llm_response(response)

        assert len(parsed) == 1
        assert parsed[0]["question"] == "What is Python?"

    @pytest.mark.asyncio
    async def test_parse_json_in_markdown(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
        )

        response = """Here is the result:
```json
[
    {
        "question": "What is AI?",
        "expected_answer": "AI is artificial intelligence.",
        "difficulty": "medium",
        "question_type": "factual"
    }
]
```
"""

        parsed = service._parse_llm_response(response)

        assert len(parsed) == 1
        assert parsed[0]["question"] == "What is AI?"

    @pytest.mark.asyncio
    async def test_parse_single_object(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test parsing single object response."""
        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
        )

        response = json.dumps(
            {
                "question": "What is ML?",
                "expected_answer": "ML is machine learning.",
                "difficulty": "hard",
                "question_type": "factual",
            }
        )

        parsed = service._parse_llm_response(response)

        assert len(parsed) == 1
        assert parsed[0]["question"] == "What is ML?"


class TestGenerateFromChunk:
    """Tests for generating test cases from chunks."""

    @pytest.mark.asyncio
    async def test_generate_from_chunk_success(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
        mock_quality_gate: MagicMock,
    ) -> None:
        """Test successful generation from a chunk."""
        # Setup mock LLM response
        mock_llm_service.completion.return_value = LLMCompletionResponse(
            content=json.dumps(
                [
                    {
                        "question": "What is the main topic?",
                        "expected_answer": "The main topic is testing.",
                        "difficulty": "easy",
                        "question_type": "factual",
                    }
                ]
            ),
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="gpt-4o-mini",
            provider="openai",
            latency_seconds=1.0,
        )

        # Setup mock quality gate - pass validation
        mock_quality_gate.validate.return_value = QualityResult(
            passed=True,
            score=0.85,
            details={},
        )

        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
            quality_gate=mock_quality_gate,
        )

        chunk = {
            "text": "This is the content about testing.",
            "source_file": "test.txt",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100,
        }

        config = GenerationConfig(skip_semantic_check=True)

        generated, rejected = await service._generate_from_chunk(
            chunk=chunk,
            count=1,
            config=config,
            kb_id=uuid.uuid4(),
        )

        assert len(generated) == 1
        assert rejected == 0
        assert generated[0].question == "What is the main topic?"
        assert generated[0].quality_score == 0.85

    @pytest.mark.asyncio
    async def test_generate_from_chunk_with_rejection(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
        mock_quality_gate: MagicMock,
    ) -> None:
        """Test generation with some rejections."""
        # Setup mock LLM response with multiple questions
        mock_llm_service.completion.return_value = LLMCompletionResponse(
            content=json.dumps(
                [
                    {
                        "question": "Good question here?",
                        "expected_answer": "Good answer.",
                        "difficulty": "easy",
                        "question_type": "factual",
                    },
                    {
                        "question": "Bad?",  # Too short
                        "expected_answer": "Bad answer.",
                        "difficulty": "easy",
                        "question_type": "factual",
                    },
                ]
            ),
            usage=TokenUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200),
            model="gpt-4o-mini",
            provider="openai",
            latency_seconds=1.0,
        )

        # First passes, second fails
        mock_quality_gate.validate.side_effect = [
            QualityResult(passed=True, score=0.85, details={}),
            QualityResult(passed=False, score=0.0, rejection_reason="Question too short"),
        ]

        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
            quality_gate=mock_quality_gate,
        )

        chunk = {
            "text": "Content for testing.",
            "source_file": "test.txt",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 100,
        }

        config = GenerationConfig(skip_semantic_check=True)

        generated, rejected = await service._generate_from_chunk(
            chunk=chunk,
            count=2,
            config=config,
            kb_id=uuid.uuid4(),
        )

        assert len(generated) == 1
        assert rejected == 1


class TestGeneratedTestCaseProvenance:
    """Tests for provenance tracking in generated test cases."""

    @pytest.mark.asyncio
    async def test_provenance_includes_source_info(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
        mock_quality_gate: MagicMock,
    ) -> None:
        """Test that provenance includes source information."""
        mock_llm_service.completion.return_value = LLMCompletionResponse(
            content=json.dumps(
                [
                    {
                        "question": "What is the topic?",
                        "expected_answer": "The topic is AI.",
                        "difficulty": "medium",
                        "question_type": "factual",
                    }
                ]
            ),
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="gpt-4o-mini",
            provider="openai",
            latency_seconds=1.0,
        )

        mock_quality_gate.validate.return_value = QualityResult(
            passed=True,
            score=0.9,
            details={"test": "data"},
        )

        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
            quality_gate=mock_quality_gate,
        )

        kb_id = uuid.uuid4()
        chunk = {
            "text": "Content about AI.",
            "source_file": "ai_guide.pdf",
            "chunk_index": 5,
            "start_char": 1000,
            "end_char": 2000,
        }

        config = GenerationConfig(
            llm_model="gpt-4o",
            llm_provider="openai",
            skip_semantic_check=True,
        )

        generated, _ = await service._generate_from_chunk(
            chunk=chunk,
            count=1,
            config=config,
            kb_id=kb_id,
        )

        assert len(generated) == 1
        provenance = generated[0].provenance

        # Check provenance fields
        assert provenance["source_file"] == "ai_guide.pdf"
        assert provenance["chunk_index"] == 5
        assert provenance["start_char"] == 1000
        assert provenance["end_char"] == 2000
        assert provenance["generation_model"] == "gpt-4o"
        assert provenance["generation_provider"] == "openai"
        assert provenance["kb_id"] == str(kb_id)
        assert "generated_at" in provenance
        assert "validation_details" in provenance


class TestCancellation:
    """Tests for generation cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_stops_generation(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test that cancel stops the generation loop."""
        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
        )

        # Set cancelled flag
        service.cancel()

        assert service._cancelled is True


class TestTokenUsageTracking:
    """Tests for token usage tracking."""

    @pytest.mark.asyncio
    async def test_total_usage_accumulates(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
        mock_quality_gate: MagicMock,
    ) -> None:
        """Test that token usage accumulates across calls."""
        mock_llm_service.completion.return_value = LLMCompletionResponse(
            content=json.dumps(
                [
                    {
                        "question": "Q?",
                        "expected_answer": "A",
                        "difficulty": "easy",
                        "question_type": "factual",
                    }
                ]
            ),
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="gpt-4o-mini",
            provider="openai",
            latency_seconds=1.0,
        )

        mock_quality_gate.validate.return_value = QualityResult(passed=True, score=0.8, details={})

        service = TestGeneratorService(
            db=db_session,
            llm_service=mock_llm_service,
            quality_gate=mock_quality_gate,
        )

        chunk = {
            "text": "Test",
            "source_file": "test.txt",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 4,
        }
        config = GenerationConfig(skip_semantic_check=True)

        # Generate twice
        await service._generate_from_chunk(chunk, 1, config, None)
        await service._generate_from_chunk(chunk, 1, config, None)

        usage = service.get_total_usage()

        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 100
        assert usage.total_tokens == 300


class TestFactoryFunction:
    """Tests for the factory function."""

    @pytest.mark.asyncio
    async def test_get_test_generator_service(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Test factory function creates service."""
        service = get_test_generator_service(db=db_session)

        assert isinstance(service, TestGeneratorService)
        assert service.db is db_session

    @pytest.mark.asyncio
    async def test_get_test_generator_service_with_dependencies(
        self,
        db_session: AsyncSession,
        mock_llm_service: MagicMock,
        mock_artifact_store: MagicMock,
        mock_quality_gate: MagicMock,
    ) -> None:
        """Test factory function with custom dependencies."""
        service = get_test_generator_service(
            db=db_session,
            llm_service=mock_llm_service,
            artifact_store=mock_artifact_store,
            quality_gate=mock_quality_gate,
        )

        assert service._llm_service is mock_llm_service
        assert service._artifact_store is mock_artifact_store
        assert service._quality_gate is mock_quality_gate
