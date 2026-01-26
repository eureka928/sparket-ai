"""Render the validator SQLAlchemy metadata into PostgreSQL DDL."""

from __future__ import annotations

from typing import List

from sqlalchemy import create_mock_engine
from sqlalchemy.dialects import postgresql

from schema import metadata


def render_ddl() -> str:
    """Return CREATE TYPE and CREATE TABLE statements ready for Draw.io."""

    statements: List[str] = []

    def capture(sql, *multiparams, **params) -> None:  # type: ignore[unused-argument]
        compiled = sql.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
        text = str(compiled).strip()
        if text:
            statements.append(text)

    engine = create_mock_engine("postgresql://", executor=capture)
    metadata.create_all(engine)

    return ";\n\n".join(statements) + ";"


def main() -> None:
    print(render_ddl())


if __name__ == "__main__":
    main()

