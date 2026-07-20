"""Test set and test case Pydantic schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.base import BaseResponseSchema, BaseSchema, PaginatedResponse


class TestCaseBase(BaseSchema):
    """Base test case schema."""

    question: str = Field(min_length=1, description="Test question")
    expected_answer: str = Field(min_length=1, description="Expected answer")
    ground_truth_context: list[str] = Field(
        default_factory=list, description="Ground truth context chunks"
    )
    difficulty: str = Field(
        default="medium",
        pattern="^(easy|medium|hard)$",
        description="Question difficulty",
    )
    category: str | None = Field(default=None, max_length=100, description="Category")
    question_type: str = Field(
        default="factual",
        pattern="^(factual|reasoning|comparison|multi_hop)$",
        description="Question type",
    )


class TestCaseCreate(TestCaseBase):
    """Schema for creating a test case."""

    template_id: UUID | None = Field(default=None, description="Template used for generation")


class TestCaseUpdate(BaseSchema):
    """Schema for updating a test case."""

    question: str | None = Field(default=None, min_length=1, description="Test question")
    expected_answer: str | None = Field(default=None, min_length=1, description="Expected answer")
    ground_truth_context: list[str] | None = Field(
        default=None, description="Ground truth context chunks"
    )
    difficulty: str | None = Field(
        default=None,
        pattern="^(easy|medium|hard)$",
        description="Question difficulty",
    )
    category: str | None = Field(default=None, max_length=100, description="Category")
    question_type: str | None = Field(
        default=None,
        pattern="^(factual|reasoning|comparison|multi_hop)$",
        description="Question type",
    )
    is_reviewed: bool | None = Field(default=None, description="Review status")


class TestCaseResponse(TestCaseBase, BaseResponseSchema):
    """Schema for test case response."""

    test_set_id: UUID = Field(description="Parent test set ID")
    template_id: UUID | None = Field(default=None, description="Template used for generation")
    is_generated: bool = Field(default=False, description="Whether LLM-generated")
    is_reviewed: bool = Field(default=False, description="Whether reviewed by human")
    quality_score: float | None = Field(
        default=None, ge=0, le=1, description="Quality score from validation"
    )
    provenance_artifact_id: UUID | None = Field(
        default=None, description="Artifact with generation provenance"
    )


class TestCaseBulkCreate(BaseSchema):
    """Schema for bulk creating test cases."""

    test_cases: list[TestCaseCreate] = Field(min_length=1, description="Test cases to create")


class TestCaseBulkReview(BaseSchema):
    """Schema for bulk reviewing test cases."""

    test_case_ids: list[UUID] = Field(min_length=1, description="Test case IDs")
    action: str = Field(pattern="^(approve|reject)$", description="Review action")


class TestSetBase(BaseSchema):
    """Base test set schema."""

    name: str = Field(min_length=1, max_length=255, description="Test set name")
    description: str | None = Field(default=None, description="Description")
    tags: list[str] = Field(default_factory=list, description="Tags")


class TestSetCreate(TestSetBase):
    """Schema for creating a test set."""

    pass


class TestSetUpdate(BaseSchema):
    """Schema for updating a test set."""

    name: str | None = Field(
        default=None, min_length=1, max_length=255, description="Test set name"
    )
    description: str | None = Field(default=None, description="Description")
    tags: list[str] | None = Field(default=None, description="Tags")


class TestSetResponse(TestSetBase, BaseResponseSchema):
    """Schema for test set response."""

    project_id: UUID = Field(description="Parent project ID")
    test_case_count: int = Field(default=0, description="Number of test cases")


class TestSetWithCases(TestSetResponse):
    """Test set response including test cases."""

    test_cases: list[TestCaseResponse] = Field(
        default_factory=list, description="Test cases in this set"
    )


class TestSetSummary(BaseSchema):
    """Minimal test set info for lists and references."""

    id: UUID
    name: str
    test_case_count: int
    created_at: datetime


class TestSetList(PaginatedResponse):
    """Paginated list of test sets."""

    items: list[TestSetResponse]


class TestSetImport(BaseSchema):
    """Schema for importing a test set from JSON."""

    name: str = Field(min_length=1, max_length=255, description="Test set name")
    description: str | None = Field(default=None, description="Description")
    tags: list[str] = Field(default_factory=list, description="Tags")
    test_cases: list[TestCaseCreate] = Field(min_length=1, description="Test cases to import")


class TestSetExport(BaseSchema):
    """Schema for exported test set."""

    id: UUID
    name: str
    description: str | None
    tags: list[str]
    created_at: datetime
    test_cases: list[dict[str, Any]] = Field(description="Exported test cases")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Export metadata")


# =============================================================================
# Test Generation Schemas
# =============================================================================


class TestGenerationConfig(BaseSchema):
    """Configuration for test case generation."""

    knowledge_base_id: UUID = Field(description="Knowledge base to generate from")
    target_count: int = Field(
        default=20, ge=1, le=500, description="Number of test cases to generate"
    )
    questions_per_chunk: int = Field(
        default=2, ge=1, le=10, description="Questions per document chunk"
    )
    difficulty_distribution: dict[str, float] | None = Field(
        default=None,
        description="Difficulty distribution (e.g., {'easy': 0.3, 'medium': 0.5, 'hard': 0.2})",
    )
    template_ids: list[UUID] | None = Field(
        default=None, description="Template IDs to use for generation"
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model for generation")
    llm_provider: str = Field(default="openai", description="LLM provider for generation")
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model for semantic duplicate check; auto-derived from llm_provider if omitted",
    )
    embedding_provider: str | None = Field(
        default=None,
        description="Embedding provider; auto-derived from llm_provider if omitted",
    )
    skip_semantic_check: bool = Field(default=False, description="Skip semantic duplicate checking")


class TestGenerationJobResponse(BaseResponseSchema):
    """Response for test generation job."""

    test_set_id: UUID = Field(description="Test set being populated")
    knowledge_base_id: UUID | None = Field(description="Source knowledge base")
    status: str = Field(description="Job status (pending, running, completed, failed, cancelled)")
    config: dict[str, Any] = Field(default_factory=dict, description="Generation configuration")
    questions_generated: int = Field(default=0, description="Questions generated so far")
    questions_total: int = Field(default=0, description="Target question count")
    questions_rejected: int = Field(default=0, description="Questions rejected by quality gates")
    started_at: datetime | None = Field(default=None, description="When generation started")
    completed_at: datetime | None = Field(default=None, description="When generation completed")
    error_message: str | None = Field(default=None, description="Error message if failed")


class TestGenerationStatusResponse(BaseSchema):
    """Response for generation status query."""

    job_id: UUID = Field(description="Generation job ID")
    status: str = Field(description="Job status")
    progress: float = Field(ge=0, le=1, description="Progress (0-1)")
    questions_generated: int = Field(description="Questions generated")
    questions_total: int = Field(description="Target question count")
    questions_rejected: int = Field(description="Questions rejected")
    started_at: datetime | None = Field(description="Start time")
    completed_at: datetime | None = Field(description="Completion time")
    error_message: str | None = Field(description="Error if failed")
