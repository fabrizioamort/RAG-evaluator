"""Add eval_judge_provider to evaluations

Revision ID: 20260603_000002
Revises: 20260603_000001
Create Date: 2026-06-03
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20260603_000002"
down_revision: Union[str, None] = "20260603_000001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("eval_judge_provider", sa.String(length=50), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.drop_column("eval_judge_provider")
