"""Test template Pydantic schemas."""

from pydantic import Field

from app.schemas.base import BaseResponseSchema, BaseSchema, PaginatedResponse


class TestTemplateBase(BaseSchema):
    """Base test template schema."""

    name: str = Field(min_length=1, max_length=255, description="Template name")
    description: str | None = Field(default=None, description="Description")
    category: str | None = Field(default=None, max_length=100, description="Template category")
    question_template: str = Field(min_length=1, description="Template for generating questions")
    answer_template: str | None = Field(default=None, description="Template for expected answers")
    entity_types: list[str] = Field(
        default_factory=list, description="Entity types this template targets"
    )
    complexity_level: str = Field(
        default="medium",
        pattern="^(easy|medium|hard)$",
        description="Complexity level",
    )


class TestTemplateCreate(TestTemplateBase):
    """Schema for creating a test template."""

    pass


class TestTemplateUpdate(BaseSchema):
    """Schema for updating a test template."""

    name: str | None = Field(
        default=None, min_length=1, max_length=255, description="Template name"
    )
    description: str | None = Field(default=None, description="Description")
    category: str | None = Field(default=None, max_length=100, description="Category")
    question_template: str | None = Field(
        default=None, min_length=1, description="Question template"
    )
    answer_template: str | None = Field(default=None, description="Answer template")
    entity_types: list[str] | None = Field(default=None, description="Entity types")
    complexity_level: str | None = Field(
        default=None,
        pattern="^(easy|medium|hard)$",
        description="Complexity level",
    )


class TestTemplateResponse(TestTemplateBase, BaseResponseSchema):
    """Schema for test template response."""

    is_builtin: bool = Field(default=False, description="Whether this is a builtin")


class TestTemplateList(PaginatedResponse):
    """Paginated list of test templates."""

    items: list[TestTemplateResponse]


class TestTemplateSummary(BaseSchema):
    """Minimal template info for selection."""

    id: str
    name: str
    category: str | None
    complexity_level: str
    is_builtin: bool
