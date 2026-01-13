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
