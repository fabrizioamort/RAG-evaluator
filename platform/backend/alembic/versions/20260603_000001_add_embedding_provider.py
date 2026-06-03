"""Add embedding_provider and embedding_base_url to rag_configs

Revision ID: 20260603_000001
Revises: 20260601_000001
Create Date: 2026-06-03
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20260603_000001"
down_revision: Union[str, None] = "20260601_000001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("rag_configs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "embedding_provider",
                sa.String(length=50),
                nullable=False,
                server_default="openai",
            )
        )
        batch_op.add_column(
            sa.Column("embedding_base_url", sa.String(length=500), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("rag_configs", schema=None) as batch_op:
        batch_op.drop_column("embedding_base_url")
        batch_op.drop_column("embedding_provider")
