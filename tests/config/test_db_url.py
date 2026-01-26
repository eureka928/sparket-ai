from types import SimpleNamespace

from sparket.config.db_url import build_database_url, ensure_env_database_url, ensure_config_database_url


def test_build_database_url_with_password():
    url = build_database_url(
        user="user",
        password="pass",
        host="localhost",
        port="5432",
        name="db",
    )
    assert url == "postgresql+asyncpg://user:pass@localhost:5432/db"


def test_build_database_url_without_password():
    url = build_database_url(
        user="user",
        password="",
        host="localhost",
        port="5432",
        name="db",
    )
    assert url == "postgresql+asyncpg://user@localhost:5432/db"


def test_ensure_env_database_url_composes(monkeypatch):
    monkeypatch.delenv("SPARKET_DATABASE__URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SPARKET_DATABASE__USER", "user")
    monkeypatch.setenv("SPARKET_DATABASE__PASSWORD", "pass")
    monkeypatch.setenv("SPARKET_DATABASE__HOST", "localhost")
    monkeypatch.setenv("SPARKET_DATABASE__PORT", "5432")
    monkeypatch.setenv("SPARKET_DATABASE__NAME", "db")

    result = ensure_env_database_url()
    assert result["composed"] is True
    assert (
        "postgresql+asyncpg://user:pass@localhost:5432/db"
        == __import__("os").environ["SPARKET_DATABASE__URL"]
    )


def test_ensure_config_database_url_composes():
    core_db = SimpleNamespace(
        url="",
        host="localhost",
        port=5432,
        user="user",
        password="pass",
        name="db",
    )

    result = ensure_config_database_url(core_db)
    assert result["composed"] is True
    assert core_db.url == "postgresql+asyncpg://user:pass@localhost:5432/db"
