"""Quality validation for generated test cases.

This service validates generated test cases against quality criteria:
- Exact duplicate detection
- Semantic duplicate detection
- Answerability validation
- Length and format checks
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.services.llm_provider import LLMProviderService
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QualityResult:
    """Result of quality validation."""

    passed: bool
    score: float
    rejection_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityConfig:
    """Configuration for quality gates."""

    # Semantic duplicate threshold (cosine similarity)
    semantic_duplicate_threshold: float = 0.92

    # Minimum question length (words)
    min_question_words: int = 4

    # Maximum question length (words)
    max_question_words: int = 100

    # Minimum answer length (words)
    min_answer_words: int = 2

    # Answerability overlap threshold (portion of answer terms in context)
    answerability_threshold: float = 0.25


class TestQualityGateService:
    """Validate generated test cases for quality.

    This service implements multiple quality gates:
    1. Exact duplicate detection - prevents identical questions
    2. Semantic duplicate detection - prevents semantically similar questions
    3. Answerability validation - ensures answer is derivable from context
    4. Length checks - ensures questions/answers meet length requirements
    """

    def __init__(
        self,
        llm_service: LLMProviderService | None = None,
        config: QualityConfig | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
    ) -> None:
        """Initialize quality gate service.

        Args:
            llm_service: LLM service for embeddings (lazy-loaded if not provided)
            config: Quality configuration settings
            embedding_model: Embedding model id used for semantic duplicate check.
                Defaults to `text-embedding-3-small` for backwards compatibility.
            embedding_provider: Provider that serves the embedding model. Defaults
                to `openai`.
        """
        self._llm_service = llm_service
        self.config = config or QualityConfig()

        # State for tracking existing questions
        self._existing_questions: set[str] = set()
        self._existing_embeddings: list[np.ndarray] = []
        self._embedding_model = embedding_model or "text-embedding-3-small"
        self._embedding_provider = embedding_provider or "openai"

    @property
    def llm_service(self) -> LLMProviderService:
        """Lazy-load LLM service."""
        if self._llm_service is None:
            self._llm_service = LLMProviderService()
        return self._llm_service

    def _normalize_question(self, question: str) -> str:
        """Normalize a question for comparison.

        Args:
            question: Original question text.

        Returns:
            Normalized question (lowercase, stripped, no punctuation trailing).
        """
        return question.strip().lower().rstrip("?").strip()

    def _check_exact_duplicate(self, question: str) -> tuple[bool, str | None]:
        """Check for exact duplicate questions.

        Args:
            question: Question to check.

        Returns:
            Tuple of (is_duplicate, reason).
        """
        normalized = self._normalize_question(question)
        if normalized in self._existing_questions:
            return True, "Exact duplicate question"
        return False, None

    def _check_length(self, question: str, expected_answer: str) -> tuple[bool, str | None]:
        """Check if question and answer meet length requirements.

        Args:
            question: Question text.
            expected_answer: Expected answer text.

        Returns:
            Tuple of (is_valid, reason if invalid).
        """
        question_words = len(question.split())
        answer_words = len(expected_answer.split())

        if question_words < self.config.min_question_words:
            return (
                False,
                f"Question too short ({question_words} words, min: {self.config.min_question_words})",
            )

        if question_words > self.config.max_question_words:
            return (
                False,
                f"Question too long ({question_words} words, max: {self.config.max_question_words})",
            )

        if answer_words < self.config.min_answer_words:
            return (
                False,
                f"Answer too short ({answer_words} words, min: {self.config.min_answer_words})",
            )

        return True, None

    def _check_answerability(
        self, expected_answer: str, context: list[str]
    ) -> tuple[bool, float, str | None]:
        """Check if the answer is derivable from the context.

        Uses a simple heuristic: checks overlap between answer terms and context.
        More sophisticated checks could use NLI models.

        Args:
            expected_answer: Expected answer text.
            context: List of context chunks.

        Returns:
            Tuple of (is_answerable, overlap_score, reason if not answerable).
        """
        if not context:
            return False, 0.0, "No context provided"

        # Combine all context
        context_text = " ".join(context).lower()
        context_words = set(context_text.split())

        # Get significant words from answer (skip very common words)
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "and",
            "but",
            "or",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "not",
            "only",
            "also",
            "just",
        }

        answer_words = set(expected_answer.lower().split()) - stop_words

        if not answer_words:
            # If no significant words, consider it answerable (trivial answer)
            return True, 1.0, None

        # Calculate overlap
        overlap = len(answer_words & context_words)
        overlap_ratio = overlap / len(answer_words)

        if overlap_ratio < self.config.answerability_threshold:
            return (
                False,
                overlap_ratio,
                f"Answer may not be derivable from context (overlap: {overlap_ratio:.2%})",
            )

        return True, overlap_ratio, None

    async def _check_semantic_duplicate(
        self, question: str, question_embedding: np.ndarray | None = None
    ) -> tuple[bool, float, str | None]:
        """Check for semantically similar existing questions.

        Args:
            question: Question to check.
            question_embedding: Pre-computed embedding (optional).

        Returns:
            Tuple of (is_duplicate, max_similarity, reason if duplicate).
        """
        if not self._existing_embeddings:
            return False, 0.0, None

        # Get embedding if not provided
        if question_embedding is None:
            try:
                response = await self.llm_service.get_embedding(
                    model=self._embedding_model,
                    input_text=question,
                    provider=self._embedding_provider,
                )
                question_embedding = np.array(response.embedding)
            except Exception as e:
                logger.warning(
                    "Failed to get embedding for semantic duplicate check",
                    error=str(e),
                )
                # Fall back to not checking semantic similarity
                return False, 0.0, None

        # Calculate cosine similarity with all existing embeddings
        max_similarity = 0.0
        for existing_emb in self._existing_embeddings:
            # Cosine similarity
            dot_product = np.dot(question_embedding, existing_emb)
            norm_product = np.linalg.norm(question_embedding) * np.linalg.norm(existing_emb)
            if norm_product > 0:
                similarity = dot_product / norm_product
                max_similarity = max(max_similarity, float(similarity))

        if max_similarity > self.config.semantic_duplicate_threshold:
            return (
                True,
                max_similarity,
                f"Semantic duplicate (similarity: {max_similarity:.2%})",
            )

        return False, max_similarity, None

    async def validate(
        self,
        question: str,
        expected_answer: str,
        context: list[str],
        skip_semantic_check: bool = False,
    ) -> QualityResult:
        """Validate a generated test case.

        Runs all quality gates and returns the result.

        Args:
            question: Generated question.
            expected_answer: Expected answer.
            context: Context chunks from which the question was generated.
            skip_semantic_check: Skip semantic duplicate check (for performance).

        Returns:
            QualityResult with validation outcome.
        """
        details: dict[str, Any] = {}

        # 1. Check length requirements
        length_valid, length_reason = self._check_length(question, expected_answer)
        details["length_check"] = {"passed": length_valid, "reason": length_reason}
        if not length_valid:
            return QualityResult(
                passed=False,
                score=0.0,
                rejection_reason=length_reason,
                details=details,
            )

        # 2. Check exact duplicates
        is_exact_dup, exact_reason = self._check_exact_duplicate(question)
        details["exact_duplicate_check"] = {
            "passed": not is_exact_dup,
            "reason": exact_reason,
        }
        if is_exact_dup:
            return QualityResult(
                passed=False,
                score=0.0,
                rejection_reason=exact_reason,
                details=details,
            )

        # 3. Check answerability
        is_answerable, overlap_score, answer_reason = self._check_answerability(
            expected_answer, context
        )
        details["answerability_check"] = {
            "passed": is_answerable,
            "overlap_score": overlap_score,
            "reason": answer_reason,
        }
        if not is_answerable:
            return QualityResult(
                passed=False,
                score=overlap_score,
                rejection_reason=answer_reason,
                details=details,
            )

        # 4. Check semantic duplicates (can be skipped for performance)
        if not skip_semantic_check and self._existing_embeddings:
            is_semantic_dup, similarity, semantic_reason = await self._check_semantic_duplicate(
                question
            )
            details["semantic_duplicate_check"] = {
                "passed": not is_semantic_dup,
                "max_similarity": similarity,
                "reason": semantic_reason,
            }
            if is_semantic_dup:
                return QualityResult(
                    passed=False,
                    score=similarity,
                    rejection_reason=semantic_reason,
                    details=details,
                )

        # All checks passed - add to tracking
        normalized = self._normalize_question(question)
        self._existing_questions.add(normalized)

        # Store embedding for future semantic checks
        if not skip_semantic_check:
            try:
                response = await self.llm_service.get_embedding(
                    model=self._embedding_model,
                    input_text=question,
                    provider=self._embedding_provider,
                )
                self._existing_embeddings.append(np.array(response.embedding))
            except Exception as e:
                logger.warning("Failed to store embedding", error=str(e))

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            question, expected_answer, context, overlap_score
        )

        logger.debug(
            "Test case passed quality gates",
            question=question[:50] + "...",
            quality_score=quality_score,
        )

        return QualityResult(
            passed=True,
            score=quality_score,
            details=details,
        )

    def _calculate_quality_score(
        self,
        question: str,
        expected_answer: str,
        context: list[str],
        answerability_score: float,
    ) -> float:
        """Calculate an overall quality score for the test case.

        Considers:
        - Question length (not too short, not too long)
        - Answer length
        - Answerability score
        - Context relevance

        Args:
            question: Question text.
            expected_answer: Expected answer text.
            context: Context chunks.
            answerability_score: Score from answerability check.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        scores = []

        # Question length score (prefer moderate length)
        q_words = len(question.split())
        if 5 <= q_words <= 20:
            scores.append(1.0)
        elif q_words < 5:
            scores.append(q_words / 5)
        else:
            scores.append(max(0.5, 1.0 - (q_words - 20) / 80))

        # Answer length score (prefer substantial but not too long)
        a_words = len(expected_answer.split())
        if 5 <= a_words <= 50:
            scores.append(1.0)
        elif a_words < 5:
            scores.append(a_words / 5)
        else:
            scores.append(max(0.5, 1.0 - (a_words - 50) / 100))

        # Answerability score (already calculated)
        scores.append(min(1.0, answerability_score * 1.5))  # Boost slightly

        # Average all scores
        return sum(scores) / len(scores) if scores else 0.5

    def add_existing_question(
        self,
        question: str,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Add an existing question to the tracking state.

        Call this for questions already in the test set to prevent duplicates.

        Args:
            question: Existing question text.
            embedding: Pre-computed embedding (optional).
        """
        normalized = self._normalize_question(question)
        self._existing_questions.add(normalized)

        if embedding is not None:
            self._existing_embeddings.append(embedding)

    async def add_existing_questions(self, questions: list[str]) -> None:
        """Add multiple existing questions to the tracking state.

        Args:
            questions: List of existing questions.
        """
        for question in questions:
            normalized = self._normalize_question(question)
            self._existing_questions.add(normalized)

            # Get embeddings for semantic duplicate detection
            try:
                response = await self.llm_service.get_embedding(
                    model=self._embedding_model,
                    input_text=question,
                    provider=self._embedding_provider,
                )
                self._existing_embeddings.append(np.array(response.embedding))
            except Exception as e:
                logger.warning("Failed to get embedding for existing question", error=str(e))

    def reset(self) -> None:
        """Reset the tracking state.

        Call this when starting generation for a new test set.
        """
        self._existing_questions.clear()
        self._existing_embeddings.clear()
        logger.debug("Quality gate state reset")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the quality gate state.

        Returns:
            Dictionary with state statistics.
        """
        return {
            "tracked_questions": len(self._existing_questions),
            "tracked_embeddings": len(self._existing_embeddings),
            "config": {
                "semantic_threshold": self.config.semantic_duplicate_threshold,
                "min_question_words": self.config.min_question_words,
                "max_question_words": self.config.max_question_words,
                "min_answer_words": self.config.min_answer_words,
                "answerability_threshold": self.config.answerability_threshold,
            },
        }


# Factory function
def get_quality_gate_service(
    llm_service: LLMProviderService | None = None,
    config: QualityConfig | None = None,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
) -> TestQualityGateService:
    """Create a quality gate service instance.

    Note: Unlike other services, this is NOT a singleton because
    each test generation job needs its own tracking state.

    Args:
        llm_service: Optional LLM service.
        config: Optional quality configuration.
        embedding_model: Optional embedding model id (default: text-embedding-3-small).
        embedding_provider: Optional embedding provider (default: openai).

    Returns:
        New TestQualityGateService instance.
    """
    return TestQualityGateService(
        llm_service=llm_service,
        config=config,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
    )
