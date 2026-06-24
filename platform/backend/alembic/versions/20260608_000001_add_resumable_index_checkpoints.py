"""Add resumable index checkpoints

Revision ID: 20260608_000001
Revises: 20260603_000002
Create Date: 2026-06-08
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20260608_000001"
down_revision: Union[str, None] = "20260603_000002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _column_exists(table_name: str, column_name: str) -> bool:
    if not _table_exists(table_name):
        return False
    return column_name in {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)
    }


def _index_exists(table_name: str, index_name: str) -> bool:
    if not _table_exists(table_name):
        return False
    return index_name in {
        index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)
    }


def upgrade() -> None:
    index_columns = {
        "progress_current": sa.Column(
            "progress_current", sa.Integer(), nullable=False, server_default="0"
        ),
        "progress_total": sa.Column(
            "progress_total", sa.Integer(), nullable=False, server_default="0"
        ),
        "last_heartbeat_at": sa.Column("last_heartbeat_at", sa.DateTime(timezone=True)),
        "resume_metadata": sa.Column("resume_metadata", sa.JSON()),
    }
    missing_index_columns = [
        column
        for name, column in index_columns.items()
        if not _column_exists("knowledge_base_indexes", name)
    ]

    if missing_index_columns:
        with op.batch_alter_table("knowledge_base_indexes", schema=None) as batch_op:
            for column in missing_index_columns:
                batch_op.add_column(column)

    if not _table_exists("knowledge_base_index_documents"):
        op.create_table(
            "knowledge_base_index_documents",
            sa.Column("index_id", sa.UUID(), nullable=False),
            sa.Column("doc_key", sa.String(length=128), nullable=False),
            sa.Column("source_path", sa.String(length=1000), nullable=False),
            sa.Column("checksum", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error_message", sa.Text()),
            sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("completed_chunks", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("started_at", sa.DateTime(timezone=True)),
            sa.Column("completed_at", sa.DateTime(timezone=True)),
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["index_id"], ["knowledge_base_indexes.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("index_id", "doc_key", name="uq_kbi_doc_index_doc_key"),
        )

    if not _index_exists("knowledge_base_index_documents", "idx_kbi_doc_index_status"):
        op.create_index(
            "idx_kbi_doc_index_status",
            "knowledge_base_index_documents",
            ["index_id", "status"],
        )

    if not _table_exists("knowledge_base_index_chunks"):
        op.create_table(
            "knowledge_base_index_chunks",
            sa.Column("index_id", sa.UUID(), nullable=False),
            sa.Column("document_id", sa.UUID(), nullable=False),
            sa.Column("doc_key", sa.String(length=128), nullable=False),
            sa.Column("chunk_hash", sa.String(length=64), nullable=False),
            sa.Column("storage_id", sa.String(length=128), nullable=False),
            sa.Column("chunk_index", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("token_usage", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error_message", sa.Text()),
            sa.Column("started_at", sa.DateTime(timezone=True)),
            sa.Column("completed_at", sa.DateTime(timezone=True)),
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["document_id"],
                ["knowledge_base_index_documents.id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["index_id"], ["knowledge_base_indexes.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("index_id", "storage_id", name="uq_kbi_chunk_index_storage_id"),
        )

    if not _index_exists("knowledge_base_index_chunks", "idx_kbi_chunk_index_status"):
        op.create_index(
            "idx_kbi_chunk_index_status",
            "knowledge_base_index_chunks",
            ["index_id", "status"],
        )
    if not _index_exists("knowledge_base_index_chunks", "idx_kbi_chunk_doc_key"):
        op.create_index(
            "idx_kbi_chunk_doc_key",
            "knowledge_base_index_chunks",
            ["index_id", "doc_key"],
        )


def downgrade() -> None:
    if _index_exists("knowledge_base_index_chunks", "idx_kbi_chunk_doc_key"):
        op.drop_index("idx_kbi_chunk_doc_key", table_name="knowledge_base_index_chunks")
    if _index_exists("knowledge_base_index_chunks", "idx_kbi_chunk_index_status"):
        op.drop_index("idx_kbi_chunk_index_status", table_name="knowledge_base_index_chunks")
    if _table_exists("knowledge_base_index_chunks"):
        op.drop_table("knowledge_base_index_chunks")
    if _index_exists("knowledge_base_index_documents", "idx_kbi_doc_index_status"):
        op.drop_index("idx_kbi_doc_index_status", table_name="knowledge_base_index_documents")
    if _table_exists("knowledge_base_index_documents"):
        op.drop_table("knowledge_base_index_documents")

    columns_to_drop = [
        name
        for name in (
            "resume_metadata",
            "last_heartbeat_at",
            "progress_total",
            "progress_current",
        )
        if _column_exists("knowledge_base_indexes", name)
    ]

    if columns_to_drop:
        with op.batch_alter_table("knowledge_base_indexes", schema=None) as batch_op:
            for column_name in columns_to_drop:
                batch_op.drop_column(column_name)
