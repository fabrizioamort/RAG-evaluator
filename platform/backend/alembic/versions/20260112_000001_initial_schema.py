"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-01-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Projects
    op.create_table(
        "projects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column(
            "tags", sa.JSON(), nullable=False, server_default="[]"
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Knowledge Bases
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("current_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("storage_path", sa.String(length=500), nullable=True),
        sa.Column("index_path", sa.String(length=500), nullable=True),
        sa.Column(
            "metadata", sa.JSON(), nullable=False, server_default="{}"
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_kb_project", "knowledge_bases", ["project_id"])
    op.create_index("idx_kb_status", "knowledge_bases", ["status"])

    # Knowledge Base Versions
    op.create_table(
        "knowledge_base_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("change_type", sa.String(length=50), nullable=False),
        sa.Column(
            "document_snapshot",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("change_description", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("knowledge_base_id", "version_number", name="uq_kb_version"),
    )
    op.create_index(
        "idx_kb_versions", "knowledge_base_versions", ["knowledge_base_id", "version_number"]
    )

    # Documents
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("checksum", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="uploaded"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_docs_kb", "documents", ["knowledge_base_id"])

    # Test Templates
    op.create_table(
        "test_templates",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("question_template", sa.Text(), nullable=False),
        sa.Column("answer_template", sa.Text(), nullable=True),
        sa.Column(
            "entity_types",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "complexity_level", sa.String(length=20), nullable=False, server_default="medium"
        ),
        sa.Column("is_builtin", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Artifacts
    op.create_table(
        "artifacts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("kind", sa.String(length=50), nullable=False),
        sa.Column("storage_key", sa.String(length=64), nullable=False),
        sa.Column(
            "content_type", sa.String(length=100), nullable=False, server_default="application/json"
        ),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_index("idx_artifacts_key", "artifacts", ["storage_key"])

    # Test Sets
    op.create_table(
        "test_sets",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "tags", sa.JSON(), nullable=False, server_default="[]"
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Test Cases
    op.create_table(
        "test_cases",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("test_set_id", sa.UUID(), nullable=False),
        sa.Column("template_id", sa.UUID(), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=False),
        sa.Column(
            "ground_truth_context",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("difficulty", sa.String(length=20), nullable=False, server_default="medium"),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("question_type", sa.String(length=50), nullable=False, server_default="factual"),
        sa.Column("is_generated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_reviewed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("provenance_artifact_id", sa.UUID(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["test_set_id"], ["test_sets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["template_id"], ["test_templates.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["provenance_artifact_id"], ["artifacts.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_test_cases_set", "test_cases", ["test_set_id"])

    # Test Generation Jobs
    op.create_table(
        "test_generation_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("test_set_id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column(
            "config", sa.JSON(), nullable=False, server_default="{}"
        ),
        sa.Column("questions_generated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("questions_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("questions_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["test_set_id"], ["test_sets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    # RAG Configs
    op.create_table(
        "rag_configs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("rag_type", sa.String(length=50), nullable=False),
        sa.Column(
            "parameters",
            sa.JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("llm_provider", sa.String(length=50), nullable=False, server_default="openai"),
        sa.Column("llm_model", sa.String(length=100), nullable=False, server_default="gpt-4o-mini"),
        sa.Column("llm_base_url", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Run Manifests
    op.create_table(
        "run_manifests",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("rag_config_snapshot", sa.JSON(), nullable=False),
        sa.Column("kb_version_snapshot", sa.JSON(), nullable=False),
        sa.Column("generation_model", sa.String(length=100), nullable=True),
        sa.Column("eval_judge_model", sa.String(length=100), nullable=True),
        sa.Column(
            "prompt_templates",
            sa.JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("rag_evaluator_version", sa.String(length=50), nullable=True),
        sa.Column("platform_version", sa.String(length=50), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Evaluations
    op.create_table(
        "evaluations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=True),
        sa.Column("kb_version_id", sa.UUID(), nullable=True),
        sa.Column("test_set_id", sa.UUID(), nullable=True),
        sa.Column("rag_config_id", sa.UUID(), nullable=True),
        sa.Column("run_manifest_id", sa.UUID(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("summary_metrics", sa.JSON(), nullable=True),
        sa.Column("cost_metrics", sa.JSON(), nullable=True),
        sa.Column("performance_metrics", sa.JSON(), nullable=True),
        sa.Column("pass_rate", sa.Float(), nullable=True),
        sa.Column("is_baseline", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("baseline_reason", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "tags", sa.JSON(), nullable=False, server_default="[]"
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["kb_version_id"], ["knowledge_base_versions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["test_set_id"], ["test_sets.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["rag_config_id"], ["rag_configs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["run_manifest_id"], ["run_manifests.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_eval_project", "evaluations", ["project_id"])
    op.create_index("idx_eval_status", "evaluations", ["status"])
    op.create_index("idx_eval_rag_config", "evaluations", ["rag_config_id"])
    op.create_index("idx_eval_completed", "evaluations", ["completed_at"])
    op.create_index(
        "idx_eval_baseline",
        "evaluations",
        ["is_baseline"],
        postgresql_where=sa.text("is_baseline = true"),
    )

    # Evaluation Jobs
    op.create_table(
        "evaluation_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("evaluation_id", sa.UUID(), nullable=False),
        sa.Column("state", sa.String(length=50), nullable=False, server_default="created"),
        sa.Column("progress_current", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_checkpoint", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "checkpoint_data",
            sa.JSON(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["evaluation_id"], ["evaluations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("evaluation_id"),
    )

    # Evaluation Results
    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("evaluation_id", sa.UUID(), nullable=False),
        sa.Column("test_case_id", sa.UUID(), nullable=True),
        sa.Column("generated_answer", sa.Text(), nullable=True),
        sa.Column("retrieved_context_artifact_id", sa.UUID(), nullable=True),
        sa.Column("retrieval_trace_artifact_id", sa.UUID(), nullable=True),
        sa.Column("raw_metrics_artifact_id", sa.UUID(), nullable=True),
        sa.Column("faithfulness_score", sa.Float(), nullable=True),
        sa.Column("faithfulness_reason", sa.Text(), nullable=True),
        sa.Column("relevancy_score", sa.Float(), nullable=True),
        sa.Column("relevancy_reason", sa.Text(), nullable=True),
        sa.Column("precision_score", sa.Float(), nullable=True),
        sa.Column("precision_reason", sa.Text(), nullable=True),
        sa.Column("recall_score", sa.Float(), nullable=True),
        sa.Column("recall_reason", sa.Text(), nullable=True),
        sa.Column("latency_seconds", sa.Float(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["evaluation_id"], ["evaluations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["test_case_id"], ["test_cases.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["retrieved_context_artifact_id"], ["artifacts.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["retrieval_trace_artifact_id"], ["artifacts.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["raw_metrics_artifact_id"], ["artifacts.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_results_eval", "evaluation_results", ["evaluation_id"])

    # Webhooks
    op.create_table(
        "webhooks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("url", sa.String(length=500), nullable=False),
        sa.Column("secret", sa.String(length=255), nullable=False),
        sa.Column(
            "events", sa.JSON(), nullable=False, server_default="[]"
        ),
        sa.Column("active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_webhooks_project", "webhooks", ["project_id"])


def downgrade() -> None:
    op.drop_table("webhooks")
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_jobs")
    op.drop_table("evaluations")
    op.drop_table("run_manifests")
    op.drop_table("rag_configs")
    op.drop_table("test_generation_jobs")
    op.drop_table("test_cases")
    op.drop_table("test_sets")
    op.drop_table("artifacts")
    op.drop_table("test_templates")
    op.drop_table("documents")
    op.drop_table("knowledge_base_versions")
    op.drop_table("knowledge_bases")
    op.drop_table("projects")
