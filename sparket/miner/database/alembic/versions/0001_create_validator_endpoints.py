"""Create validator_endpoint table

Revision ID: 0001_create_validator_endpoints
Revises:
Create Date: 2025-11-17 20:30:00
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0001_create_validator_endpoints"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "validator_endpoint" in inspector.get_table_names():
        uniques = {uc["name"] for uc in inspector.get_unique_constraints("validator_endpoint")}
        if "uq_validator_endpoint_hotkey" not in uniques:
            with op.batch_alter_table("validator_endpoint", recreate="always") as batch_op:
                batch_op.create_unique_constraint("uq_validator_endpoint_hotkey", ["hotkey"])
        return
    op.create_table(
        "validator_endpoint",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("hotkey", sa.String(length=64), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=True),
        sa.Column("port", sa.Integer(), nullable=True),
        sa.Column("url", sa.String(length=512), nullable=True),
        sa.Column("token", sa.String(length=512), nullable=True),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("hotkey", name="uq_validator_endpoint_hotkey"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "validator_endpoint" in inspector.get_table_names():
        op.drop_table("validator_endpoint")

