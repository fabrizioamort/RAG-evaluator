"""add_geval_and_metric_config

Revision ID: 56bd3526559d
Revises: c4371683d17a
Create Date: 2026-01-18 10:51:13.598030

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '56bd3526559d'
down_revision: Union[str, None] = 'c4371683d17a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add g_eval columns to evaluation_results
    with op.batch_alter_table('evaluation_results', schema=None) as batch_op:
        batch_op.add_column(sa.Column('g_eval_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('g_eval_reason', sa.Text(), nullable=True))

    # Add metric_config column to evaluations
    with op.batch_alter_table('evaluations', schema=None) as batch_op:
        batch_op.add_column(sa.Column('metric_config', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('evaluations', schema=None) as batch_op:
        batch_op.drop_column('metric_config')

    with op.batch_alter_table('evaluation_results', schema=None) as batch_op:
        batch_op.drop_column('g_eval_reason')
        batch_op.drop_column('g_eval_score')
