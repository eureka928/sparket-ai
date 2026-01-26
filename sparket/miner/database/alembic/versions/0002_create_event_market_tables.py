"""Create event and market tables for miner game data sync

Revision ID: 0002_create_event_market_tables
Revises: 0001_create_validator_endpoints
Create Date: 2025-12-16 23:20:00
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0002_create_event_market_tables"
down_revision = "0001_create_validator_endpoints"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = inspector.get_table_names()
    
    # Create event table if not exists
    if "event" not in existing_tables:
        op.create_table(
            "event",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("event_id", sa.Integer(), nullable=False, index=True),
            sa.Column("home_team", sa.String(length=128), nullable=True),
            sa.Column("away_team", sa.String(length=128), nullable=True),
            sa.Column("venue", sa.String(length=255), nullable=True),
            sa.Column("start_time_utc", sa.DateTime(), nullable=False, index=True),
            sa.Column("status", sa.String(length=32), nullable=False, default="scheduled"),
            sa.Column("league", sa.String(length=32), nullable=True),
            sa.Column("sport", sa.String(length=32), nullable=True),
            sa.Column("synced_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("event_id", name="uq_event_event_id"),
        )
    
    # Create market table if not exists
    if "market" not in existing_tables:
        op.create_table(
            "market",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("market_id", sa.Integer(), nullable=False, index=True),
            sa.Column("event_id", sa.Integer(), nullable=False, index=True),
            sa.Column("kind", sa.String(length=32), nullable=False),
            sa.Column("line", sa.Float(), nullable=True),
            sa.Column("points_team_id", sa.Integer(), nullable=True),
            sa.Column("synced_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("market_id", name="uq_market_market_id"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = inspector.get_table_names()
    
    if "market" in existing_tables:
        op.drop_table("market")
    if "event" in existing_tables:
        op.drop_table("event")

