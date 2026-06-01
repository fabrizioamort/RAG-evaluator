"""Add llm_reasoning_effort to rag_configs

Revision ID: 20260601_000001
Revises: 20260530_000001
Create Date: 2026-06-01
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20260601_000001"
down_revision: Union[str, None] = "20260530_000001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("rag_configs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("llm_reasoning_effort", sa.String(length=20), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("rag_configs", schema=None) as batch_op:
        batch_op.drop_column("llm_reasoning_effort")
