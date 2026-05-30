"""Split build and query RAG configuration

Revision ID: 20260530_000001
Revises: a1b2c3d4e5f6
Create Date: 2026-05-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260530_000001"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


json_type = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("rag_configs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "embedding_model",
                sa.String(length=100),
                nullable=False,
                server_default="text-embedding-3-small",
            )
        )

    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("query_overrides", json_type, nullable=False, server_default="{}")
        )
        batch_op.add_column(sa.Column("eval_judge_model", sa.String(length=100), nullable=True))

    with op.batch_alter_table("run_manifests", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("build_config_snapshot", json_type, nullable=False, server_default="{}")
        )
        batch_op.add_column(sa.Column("query_overrides", json_type, nullable=False, server_default="{}"))
        batch_op.add_column(
            sa.Column("effective_config_snapshot", json_type, nullable=False, server_default="{}")
        )

    # Backfill new snapshots from the legacy run snapshot where present.
    op.execute(
        "UPDATE run_manifests "
        "SET build_config_snapshot = rag_config_snapshot, "
        "effective_config_snapshot = rag_config_snapshot "
        "WHERE rag_config_snapshot IS NOT NULL"
    )


def downgrade() -> None:
    with op.batch_alter_table("run_manifests", schema=None) as batch_op:
        batch_op.drop_column("effective_config_snapshot")
        batch_op.drop_column("query_overrides")
        batch_op.drop_column("build_config_snapshot")

    with op.batch_alter_table("evaluations", schema=None) as batch_op:
        batch_op.drop_column("eval_judge_model")
        batch_op.drop_column("query_overrides")

    with op.batch_alter_table("rag_configs", schema=None) as batch_op:
        batch_op.drop_column("embedding_model")
