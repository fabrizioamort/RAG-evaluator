"""Add metadata to test cases.

Revision ID: 20260626_000001
Revises: 20260608_000001
Create Date: 2026-06-26
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260626_000001"
down_revision: Union[str, None] = "20260608_000001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("test_cases") as batch_op:
        batch_op.add_column(
            sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}")
        )


def downgrade() -> None:
    with op.batch_alter_table("test_cases") as batch_op:
        batch_op.drop_column("metadata")
