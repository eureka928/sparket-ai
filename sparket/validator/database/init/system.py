"""System-level helpers for managing local PostgreSQL services."""

from __future__ import annotations

import os
import subprocess
import tempfile
import urllib.request

from .utils import _command_exists, _is_root, _run

__all__ = [
    "_is_ubuntu",
    "_ubuntu_codename",
    "_ensure_pgdg_repo",
    "_ensure_postgres_running",
]


def _ensure_postgres_running() -> None:
    if _command_exists("pg_isready") and _run("pg_isready -q") == 0:
        return

    if _is_root() and _is_ubuntu() and _command_exists("apt"):
        _ensure_pgdg_repo()
        _run("apt update")
        _run("apt install -y postgresql-18 postgresql-client-18")
        if _command_exists("systemctl"):
            _run("systemctl enable postgresql")
            _run("systemctl start postgresql")
            if _command_exists("pg_isready") and _run("pg_isready -q") != 0:
                _run("systemctl enable postgresql@18-main")
                _run("systemctl start postgresql@18-main")


def _is_ubuntu() -> bool:
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as release_file:
            data = release_file.read().lower()
        return "id=ubuntu" in data or "ubuntu" in data
    except Exception:
        return False


def _ubuntu_codename() -> str:
    if _command_exists("lsb_release"):
        try:
            out = subprocess.check_output(["lsb_release", "-cs"], text=True).strip()
            if out:
                return out
        except Exception:
            pass
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as release_file:
            for line in release_file:
                if line.startswith("VERSION_CODENAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return "jammy"


def _ensure_pgdg_repo() -> None:
    key_path = "/usr/share/keyrings/postgresql.gpg"
    if not os.path.exists(key_path):
        try:
            with urllib.request.urlopen(
                "https://www.postgresql.org/media/keys/ACCC4CF8.asc", timeout=15
            ) as resp:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(resp.read())
                    tmp_path = tmp.name
            _run("apt install -y gnupg")
            subprocess.check_call(["gpg", "--dearmor", "-o", key_path, tmp_path])
            os.remove(tmp_path)
        except Exception:
            pass

    list_path = "/etc/apt/sources.list.d/pgdg.list"
    if not os.path.exists(list_path):
        codename = _ubuntu_codename()
        repo_line = (
            f"deb [signed-by={key_path}] http://apt.postgresql.org/pub/repos/apt {codename}-pgdg main\n"
        )
        try:
            with open(list_path, "w", encoding="utf-8") as list_file:
                list_file.write(repo_line)
        except Exception:
            pass


