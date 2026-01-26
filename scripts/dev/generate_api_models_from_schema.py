from __future__ import annotations

import inspect
from pathlib import Path

from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy.sql.schema import Column
from sqlalchemy import Integer, BigInteger, String, Numeric, Boolean, DateTime

# Import schema
from sparket.validator.database import schema as dbschema

OUT_DIR = Path("sparket/protocol/models/v1/generated")
INIT_FILE = OUT_DIR.parent / "__init__.py"
VERSION_FILE = Path("sparket/protocol/models/v1/VERSION.txt")


def _py_type(col: Column) -> str:
    t = col.type
    # very simple mapping; extend as needed
    if isinstance(t, (Integer, BigInteger)):
        return "int"
    if isinstance(t, Numeric):
        return "float"
    if isinstance(t, Boolean):
        return "bool"
    if isinstance(t, DateTime):
        return "datetime"
    if isinstance(t, String):
        return "str"
    # JSONB or others -> dict
    return "dict"


def _optional(py_t: str, col: Column) -> str:
    return f"Optional[{py_t}]" if col.nullable else py_t


def _get_decls() -> list[type]:
    decls = []
    for name, obj in vars(dbschema).items():
        if inspect.isclass(obj) and isinstance(obj, DeclarativeMeta):
            # Respect table-level __api_expose__ hints if present; otherwise skip
            hints = getattr(obj, "__api_expose__", None)
            if hints:
                decls.append(obj)
    return decls


def _render_model(cls: type, variant: str, name: str) -> str:
    # variant: "read" or "write"; name: model class name
    lines = ["from __future__ import annotations", "", "from datetime import datetime", "from typing import Optional", "from pydantic import BaseModel"]
    lines.append("")
    lines.append(f"class {name}(BaseModel):")
    for col in cls.__table__.columns:  # type: ignore[attr-defined]
        if col.info.get("api_exclude"):
            continue
        py_t = _py_type(col)
        py_t = _optional(py_t, col)
        # For write models, drop autoincrement PKs / server defaults
        if variant == "write" and (col.primary_key and col.autoincrement):
            continue
        alias = col.info.get("alias") if isinstance(col.info, dict) else None
        field_name = alias or col.name
        lines.append(f"    {field_name}: {py_t}")
    lines.append("")
    return "\n".join(lines)


def _write_file(path: Path, content: str) -> bool:
    prev = path.read_text() if path.exists() else None
    if prev == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def _bump_version():
    cur = VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "0"
    try:
        nxt = str(int(cur) + 1)
    except Exception:
        nxt = "1"
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(nxt + "\n")


def main() -> None:
    changed = False
    decls = _get_decls()
    for cls in decls:
        hints = getattr(cls, "__api_expose__", {}) or {}
        for ver, cfg in hints.items():
            if ver != "v1":
                continue
            read_name = cfg.get("read")
            write_name = cfg.get("write")
            if read_name:
                code = _render_model(cls, variant="read", name=read_name)
                changed |= _write_file(OUT_DIR / f"{read_name}.py", code)
            if write_name:
                code = _render_model(cls, variant="write", name=write_name)
                changed |= _write_file(OUT_DIR / f"{write_name}.py", code)

    # Update __init__ with simple re-exports
    exports = []
    if OUT_DIR.exists():
        for f in sorted(OUT_DIR.glob("*.py")):
            if f.name == "__init__.py":
                continue
            mod = f"sparket.protocol.models.v1.generated.{f.stem}"
            exports.append(f"from {mod} import *")
    _write_file(INIT_FILE, "\n".join(exports) + "\n")

    if changed:
        _bump_version()


if __name__ == "__main__":
    main()
