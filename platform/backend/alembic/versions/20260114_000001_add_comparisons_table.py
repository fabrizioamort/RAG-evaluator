"""Add comparisons table

Revision ID: 0002
Revises: 0001
Create Date: 2026-01-14
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Comparisons
    op.create_table(
        "comparisons",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("baseline_evaluation_id", sa.UUID(), nullable=False),
        sa.Column("compared_evaluation_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("aggregate_metrics", sa.JSON(), nullable=True),
        sa.Column("per_question_deltas", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["baseline_evaluation_id"], ["evaluations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_comparisons_project", "comparisons", ["project_id"])
    op.create_index("idx_comparisons_baseline", "comparisons", ["baseline_evaluation_id"])


def downgrade() -> None:
    op.drop_index("idx_comparisons_baseline", table_name="comparisons")
    op.drop_index("idx_comparisons_project", table_name="comparisons")
    op.drop_table("comparisons")
