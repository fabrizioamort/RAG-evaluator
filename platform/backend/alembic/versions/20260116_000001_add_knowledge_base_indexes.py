"""Add knowledge_base_indexes table and related changes

Revision ID: 0003
Revises: 0002
Create Date: 2026-01-16
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create knowledge_base_indexes table
    op.create_table(
        "knowledge_base_indexes",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("knowledge_base_id", sa.UUID(), nullable=False),
        sa.Column("kb_version_id", sa.UUID(), nullable=True),
        sa.Column("rag_config_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("physical_id", sa.String(length=64), nullable=False),
        sa.Column("storage_type", sa.String(length=50), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
        sa.Column("build_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("build_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("build_duration_seconds", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["kb_version_id"], ["knowledge_base_versions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["rag_config_id"], ["rag_configs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("physical_id"),
    )

    # Create indexes for common queries
    op.create_index("idx_kbi_knowledge_base", "knowledge_base_indexes", ["knowledge_base_id"])
    op.create_index("idx_kbi_status", "knowledge_base_indexes", ["status"])
    op.create_index("idx_kbi_physical_id", "knowledge_base_indexes", ["physical_id"])
    op.create_index("idx_kbi_rag_config", "knowledge_base_indexes", ["rag_config_id"])

    # 2. Add archived_at to knowledge_bases
    op.add_column(
        "knowledge_bases",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )

    # 3. Add knowledge_base_index_id to evaluations
    op.add_column(
        "evaluations",
        sa.Column("knowledge_base_index_id", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_evaluations_knowledge_base_index",
        "evaluations",
        "knowledge_base_indexes",
        ["knowledge_base_index_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index("idx_evaluations_kb_index", "evaluations", ["knowledge_base_index_id"])


def downgrade() -> None:
    # Remove knowledge_base_index_id from evaluations
    op.drop_index("idx_evaluations_kb_index", table_name="evaluations")
    op.drop_constraint("fk_evaluations_knowledge_base_index", "evaluations", type_="foreignkey")
    op.drop_column("evaluations", "knowledge_base_index_id")

    # Remove archived_at from knowledge_bases
    op.drop_column("knowledge_bases", "archived_at")

    # Drop knowledge_base_indexes table
    op.drop_index("idx_kbi_rag_config", table_name="knowledge_base_indexes")
    op.drop_index("idx_kbi_physical_id", table_name="knowledge_base_indexes")
    op.drop_index("idx_kbi_status", table_name="knowledge_base_indexes")
    op.drop_index("idx_kbi_knowledge_base", table_name="knowledge_base_indexes")
    op.drop_table("knowledge_base_indexes")
