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
    with op.batch_alter_table("knowledge_bases") as batch_op:
        batch_op.add_column(sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True))

    # 3. Add knowledge_base_index_id to evaluations
    with op.batch_alter_table("evaluations") as batch_op:
        batch_op.add_column(sa.Column("knowledge_base_index_id", sa.UUID(), nullable=True))
        batch_op.create_foreign_key(
            "fk_evaluations_knowledge_base_index",
            "knowledge_base_indexes",
            ["knowledge_base_index_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_index("idx_evaluations_kb_index", ["knowledge_base_index_id"])


def downgrade() -> None:
    # Remove knowledge_base_index_id from evaluations
    with op.batch_alter_table("evaluations") as batch_op:
        # Check if index exists before dropping to be safe, or just let batch handle it
        batch_op.drop_index("idx_evaluations_kb_index")
        batch_op.drop_constraint("fk_evaluations_knowledge_base_index", type_="foreignkey")
        batch_op.drop_column("knowledge_base_index_id")

    # Remove archived_at from knowledge_bases
    with op.batch_alter_table("knowledge_bases") as batch_op:
        batch_op.drop_column("archived_at")

    # Drop knowledge_base_indexes table
    op.drop_index("idx_kbi_rag_config", table_name="knowledge_base_indexes")
    op.drop_index("idx_kbi_physical_id", table_name="knowledge_base_indexes")
    op.drop_index("idx_kbi_status", table_name="knowledge_base_indexes")
    op.drop_index("idx_kbi_knowledge_base", table_name="knowledge_base_indexes")
    op.drop_table("knowledge_base_indexes")