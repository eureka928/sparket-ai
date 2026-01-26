"""Docker orchestration for validator PostgreSQL bootstrap."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import bittensor as bt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from .utils import _command_exists, _sanitize_url_for_log

__all__ = ["_ensure_docker_postgres_running"]


@dataclass(slots=True)
class DockerSettings:
    container: str
    image: str
    volume: str
    user: str
    password: str
    dbname: str
    host_port: int
    container_port: int = 5432
    url: Optional[str] = None
    sanitized_url: Optional[str] = None
    from_env_port: Optional[int] = None
    from_url_port: Optional[int] = None
    test_mode: bool = False


def _ensure_docker_postgres_running(core_db, test_mode: bool = False) -> None:
    settings = _build_settings(core_db, test_mode=test_mode)
    if settings is None:
        return

    if not _command_exists("docker"):
        bt.logging.warning({"docker_pg": {"skip": "docker_not_found"}})
        return

    _pull_image(settings.image)

    container_info = _inspect_container(settings.container)
    if container_info and not _port_binding_matches(container_info, settings.host_port):
        bt.logging.warning(
            {
                "docker_pg": {
                    "port_mismatch": True,
                    "existing": _extract_container_ports(container_info),
                    "expected": settings.host_port,
                    "action": "recreate_container",
                }
            }
        )
        _remove_container(settings.container)
        container_info = None

    if container_info is None:
        _prepare_volume_for_bootstrap(settings.volume)
        _run_container(settings)
    else:
        try:
            _ensure_container_running(
                settings.container, settings.host_port, settings.volume
            )
        except RuntimeError:
            _prepare_volume_for_bootstrap(settings.volume)
            _run_container(settings)

    if not _wait_for_port(settings.host_port, timeout=90):
        bt.logging.error(
            {"docker_pg": {"port_timeout": True, "port": settings.host_port}}
        )
        _log_container_tail(settings.container, 40)
        return

    if not _wait_for_postgres(settings):
        bt.logging.error(
            {"docker_pg": {"postgres_ready_timeout": True, "port": settings.host_port}}
        )
        _log_container_tail(settings.container, 80)


def _build_settings(core_db, test_mode: bool = False) -> Optional[DockerSettings]:
    docker_cfg = getattr(core_db, "docker", None)
    if not docker_cfg or not getattr(docker_cfg, "enabled", False):
        bt.logging.info({"docker_pg": {"skip": "docker_disabled"}})
        return None

    # Use effective methods if available for test mode, otherwise fall back to direct attrs
    if hasattr(docker_cfg, "effective_container_name"):
        container = docker_cfg.effective_container_name(test_mode)
        volume = docker_cfg.effective_volume(test_mode)
        host_port = int(docker_cfg.effective_port(test_mode))
    else:
        container = docker_cfg.container_name
        volume = docker_cfg.volume
        host_port = int(getattr(docker_cfg, "port", 5432))
    image = docker_cfg.image

    env_port = os.getenv("SPARKET_DATABASE__PORT") or os.getenv("DATABASE_PORT")
    from_env_port = None
    if env_port:
        try:
            host_port = int(env_port)
            from_env_port = host_port
        except ValueError:
            bt.logging.warning(
                {"docker_pg": {"port_from_env_invalid": env_port}}
            )

    url = getattr(core_db, "url", None) or os.getenv("SPARKET_DATABASE__URL") or os.getenv("DATABASE_URL")
    from_url_port = None
    user = getattr(core_db, "user", None)
    password = getattr(core_db, "password", None)
    dbname = getattr(core_db, "name", None)

    if url:
        bt.logging.info(
            {
                "docker_pg": {
                    "parsing_url": True,
                    "url_redacted": _sanitize_url_for_log(url),
                }
            }
        )
        try:
            from urllib.parse import urlsplit

            parts = urlsplit(url)
            if parts.port:
                host_port = parts.port
                from_url_port = parts.port
            if parts.username and not user:
                user = parts.username
            if parts.password is not None and password is None:
                password = parts.password or ""
            if parts.path and not dbname:
                dbname = parts.path.lstrip("/") or dbname
        except Exception as exc:
            bt.logging.error({"docker_pg": {"url_parse_error": str(exc)}})

    # Override dbname for test mode if test_name is available
    if test_mode and hasattr(core_db, "test_name") and core_db.test_name:
        dbname = core_db.test_name

    if not user or not dbname:
        bt.logging.error(
            {
                "docker_pg": {
                    "skip": "missing_required_credentials",
                    "user": bool(user),
                    "dbname": bool(dbname),
                }
            }
        )
        return None

    if password is None:
        password = ""

    sanitized_url = _sanitize_url_for_log(url) if url else None
    bt.logging.info(
        {
            "docker_pg": {
                "config": {
                    "container": container,
                    "image": image,
                    "volume": volume,
                    "host_port": host_port,
                    "url_port": from_url_port,
                    "env_port": from_env_port,
                    "user": user,
                    "dbname": dbname,
                    "url_redacted": sanitized_url,
                    "test_mode": test_mode,
                }
            }
        }
    )

    return DockerSettings(
        container=container,
        image=image,
        volume=volume,
        user=user,
        password=password,
        dbname=dbname,
        host_port=host_port,
        url=url,
        sanitized_url=sanitized_url,
        from_env_port=from_env_port,
        from_url_port=from_url_port,
        test_mode=test_mode,
    )


def _pull_image(image: str) -> None:
    try:
        bt.logging.info({"docker_pg": {"action": "pull", "image": image}})
        result = subprocess.run(
            ["docker", "pull", image], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            bt.logging.info({"docker_pg": {"pulled": True}})
        else:
            bt.logging.warning(
                {"docker_pg": {"pull_failed": result.stderr[:200] if result.stderr else None}}
            )
    except Exception as exc:
        bt.logging.warning({"docker_pg": {"pull_error": str(exc)}})


def _inspect_container(container: str) -> Optional[dict]:
    try:
        result = subprocess.run(
            ["docker", "inspect", container],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        import json

        return json.loads(result.stdout)[0]
    except Exception:
        return None


def _port_binding_matches(container_info: dict, expected_port: int) -> bool:
    ports = _extract_container_ports(container_info)
    return expected_port in ports


def _extract_container_ports(container_info: dict) -> set[int]:
    ports = set()
    try:
        bindings = (
            container_info.get("NetworkSettings", {})
            .get("Ports", {})
            .get("5432/tcp", [])
        )
        for binding in bindings:
            host_port = binding.get("HostPort")
            if host_port:
                ports.add(int(host_port))
    except Exception:
        pass
    return ports


def _remove_container(container: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def _prepare_volume_for_bootstrap(volume: str) -> None:
    if not volume:
        return

    if _volume_exists(volume):
        bt.logging.warning(
            {
                "docker_pg": {
                    "removing_existing_volume": True,
                    "volume": volume,
                    "reason": "ensure fresh credentials",
                }
            }
        )
        _remove_volume(volume)


def _volume_exists(volume: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "volume", "inspect", volume],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _remove_volume(volume: str) -> None:
    subprocess.run(
        ["docker", "volume", "rm", "-f", volume],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _run_container(settings: DockerSettings) -> None:
    bt.logging.info(
        {
            "docker_pg": {
                "action": "run",
                "container": settings.container,
                "image": settings.image,
                "host_port": settings.host_port,
                "container_port": settings.container_port,
                "volume": settings.volume,
                "db": settings.dbname,
                "user": settings.user,
            }
        }
    )

    if _port_in_use(settings.host_port, settings.container):
        bt.logging.warning(
            {
                "docker_pg": {
                    "port_conflict": settings.host_port,
                    "note": "Port already in use before run",
                }
            }
        )

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        settings.container,
        "-e",
        f"POSTGRES_USER={settings.user}",
        "-e",
        f"POSTGRES_PASSWORD={settings.password}",
        "-e",
        f"POSTGRES_DB={settings.dbname}",
        "-p",
        f"{settings.host_port}:{settings.container_port}",
        "-v",
        f"{settings.volume}:/var/lib/postgresql",
        "--restart",
        "unless-stopped",
        settings.image,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False, timeout=60
    )
    if result.returncode != 0:
        bt.logging.error(
            {
                "docker_pg": {
                    "start_failed": True,
                    "error": result.stderr[:500] if result.stderr else result.stdout[:500],
                }
            }
        )


def _ensure_container_running(container: str, port: int, volume: str) -> None:
    state = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    if state.returncode != 0:
        return
    if state.stdout.strip().lower() == "true":
        bt.logging.info({"docker_pg": {"already_running": True}})
        return

    start = subprocess.run(
        ["docker", "start", container],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if start.returncode == 0:
        bt.logging.info({"docker_pg": {"started_existing": True}})
        return

    error_msg = start.stderr or start.stdout or ""
    if "address already in use" in error_msg.lower():
        bt.logging.warning(
            {"docker_pg": {"restart_port_conflict": True, "action": "recreate"}}
        )
        _remove_container(container)
        if volume:
            _remove_volume(volume)
    else:
        bt.logging.error(
            {
                "docker_pg": {
                    "start_existing_failed": True,
                    "error": error_msg[:500] if error_msg else "unknown",
                }
            }
        )

    # Re-run container after removal attempts
    raise RuntimeError("container_restart_failed")


def _port_in_use(port: int, container: str) -> bool:
    docker_conflict = subprocess.run(
        ["docker", "ps", "--filter", f"publish={port}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    if docker_conflict.returncode == 0:
        names = {name.strip() for name in docker_conflict.stdout.splitlines() if name.strip()}
        for name in names:
            if name != container:
                bt.logging.warning(
                    {
                        "docker_pg": {
                            "port_conflict_docker": port,
                            "other_container": name,
                        }
                    }
                )
                return True

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            return False
    bt.logging.warning(
        {
            "docker_pg": {
                "port_conflict_system": port,
                "note": "Port already bound by system process",
            }
        }
    )
    return True


def _wait_for_port(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                bt.logging.info({"docker_pg": {"port_open": True, "port": port}})
                return True
        time.sleep(0.5)
    return False


def _wait_for_postgres(settings: DockerSettings) -> bool:
    timeout = 120.0
    interval = 2.0
    started = time.time()
    elapsed = 0.0

    while elapsed < timeout:
        if _test_direct_connection(settings, timeout=5.0):
            bt.logging.info(
                {
                    "docker_pg": {
                        "postgres_ready": True,
                        "method": "direct_connection",
                        "elapsed_seconds": round(elapsed, 1),
                    }
                }
            )
            return True
        if _pg_isready_available(settings):
            bt.logging.info(
                {
                    "docker_pg": {
                        "postgres_ready": True,
                        "method": "pg_isready",
                        "elapsed_seconds": round(elapsed, 1),
                    }
                }
            )
            return True
        time.sleep(interval)
        elapsed = time.time() - started
    return False


def _test_direct_connection(settings: DockerSettings, timeout: float) -> bool:
    test_url = (
        f"postgresql+asyncpg://{settings.user}:{settings.password}"
        f"@127.0.0.1:{settings.host_port}/postgres"
    )
    engine = create_async_engine(
        test_url,
        pool_size=1,
        max_overflow=0,
        future=True,
    )

    async def _probe() -> bool:
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
        finally:
            await engine.dispose()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(asyncio.wait_for(_probe(), timeout))
        return asyncio.run(asyncio.wait_for(_probe(), timeout))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(asyncio.wait_for(_probe(), timeout))
        finally:
            loop.close()
    except Exception:
        return False


def _pg_isready_available(settings: DockerSettings) -> bool:
    if not _command_exists("pg_isready"):
        return False
    result = subprocess.run(
        ["pg_isready", "-h", "127.0.0.1", "-p", str(settings.host_port), "-U", settings.user],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _log_container_tail(container: str, lines: int) -> None:
    try:
        logs = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if logs.returncode == 0 and logs.stdout:
            bt.logging.error(
                {
                    "docker_pg": {
                        "container_logs": logs.stdout[-2000:],
                    }
                }
            )
    except Exception:
        pass


