"""add_ledger_accumulators

Add accumulator columns to miner_rolling_score and create ledger_state table
for the primary+auditor validator model.

Revision ID: a1b2c3d4e5f6
Revises: 6eea4bc67ecd
Create Date: 2026-02-05 00:00:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '6eea4bc67ecd'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- 1. Add accumulator columns to miner_rolling_score --
    # Per outcome-verifiable metric (auditors can independently check)
    op.add_column('miner_rolling_score', sa.Column(
        'brier_ws', sa.Numeric(), nullable=True,
        comment='Weighted sum for Brier scores (decay-weighted)',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'brier_wt', sa.Numeric(), nullable=True,
        comment='Weight sum for Brier scores (sum of decay weights)',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'fq_ws', sa.Numeric(), nullable=True,
        comment='Weighted sum for FQ (= 1 - 2*brier)',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'fq_wt', sa.Numeric(), nullable=True,
        comment='Weight sum for FQ',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'pss_ws', sa.Numeric(), nullable=True,
        comment='Weighted sum for PSS',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'pss_wt', sa.Numeric(), nullable=True,
        comment='Weight sum for PSS',
    ))

    # Per CLV/CLE-derived metric (auditors trust these from checkpoint)
    op.add_column('miner_rolling_score', sa.Column(
        'es_ws', sa.Numeric(), nullable=True,
        comment='Weighted sum for CLE (economic edge)',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'es_wt', sa.Numeric(), nullable=True,
        comment='Weight sum for CLE',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'mes_ws', sa.Numeric(), nullable=True,
        comment='Weighted sum for MES (market efficiency)',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'mes_wt', sa.Numeric(), nullable=True,
        comment='Weight sum for MES',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'sos_ws', sa.Numeric(), nullable=True,
        comment='Weighted sum for SOS (originality)',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'sos_wt', sa.Numeric(), nullable=True,
        comment='Weight sum for SOS',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'lead_ws', sa.Numeric(), nullable=True,
        comment='Weighted sum for lead ratio',
    ))
    op.add_column('miner_rolling_score', sa.Column(
        'lead_wt', sa.Numeric(), nullable=True,
        comment='Weight sum for lead ratio',
    ))

    # -- 2. Create ledger_state table --
    op.create_table(
        'ledger_state',
        sa.Column('id', sa.Integer(), primary_key=True, default=1,
                  comment='Singleton row (always id=1)'),
        sa.Column('checkpoint_epoch', sa.Integer(), nullable=False, server_default='1',
                  comment='Current checkpoint epoch (incremented on recompute)'),
        sa.Column('last_checkpoint_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When the last checkpoint was exported'),
        sa.Column('last_delta_at', sa.DateTime(timezone=True), nullable=True,
                  comment='When the last delta was exported'),
        sa.Column('last_delta_id', sa.String(), nullable=True,
                  comment='ID of the last delta exported'),
    )

    # Insert default row
    op.execute("INSERT INTO ledger_state (id, checkpoint_epoch) VALUES (1, 1) ON CONFLICT DO NOTHING")


def downgrade() -> None:
    # Drop ledger_state table
    op.drop_table('ledger_state')

    # Drop accumulator columns from miner_rolling_score
    for col in (
        'lead_wt', 'lead_ws', 'sos_wt', 'sos_ws',
        'mes_wt', 'mes_ws', 'es_wt', 'es_ws',
        'pss_wt', 'pss_ws', 'fq_wt', 'fq_ws',
        'brier_wt', 'brier_ws',
    ):
        op.drop_column('miner_rolling_score', col)
