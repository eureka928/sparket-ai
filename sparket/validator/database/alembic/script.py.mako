"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
${imports if imports else ""}

_ENUM_NAMES = (
    "market_kind_enum",
    "market_result_enum",
    "price_side_enum",
    "score_component_enum",
    "task_type_enum",
)
_ENUMS = [globals().get(name) for name in _ENUM_NAMES]


def _create_enums(bind) -> None:
    for enum in (e for e in _ENUMS if e is not None):
        try:
            enum.create(bind, checkfirst=True)
        except AttributeError:
            continue


def _drop_enums(bind) -> None:
    for enum in (e for e in reversed(_ENUMS) if e is not None):
        try:
            enum.drop(bind, checkfirst=True)
        except AttributeError:
            continue


# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    bind = op.get_bind()
    _create_enums(bind)
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
    bind = op.get_bind()
    _drop_enums(bind)
