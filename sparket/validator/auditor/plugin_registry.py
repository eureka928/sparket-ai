"""Plugin registry for auditor work types.

The core auditor runtime dispatches to registered plugins via this
registry. New capabilities are added by writing a TaskHandler class
and registering it - zero changes to the core loop.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import bittensor as bt


@dataclass
class TaskResult:
    """Result from a plugin execution."""

    plugin_name: str
    plugin_version: str
    status: str  # "pass", "fail", "error", "skip"
    evidence: dict[str, Any] = field(default_factory=dict)
    attestation: str = ""
    completed_at: datetime = field(default_factory=lambda: datetime.now())


@dataclass
class AuditorContext:
    """Context passed to plugins each cycle.

    Provides everything a plugin needs: latest checkpoint, new deltas,
    accumulated state, wallet, subtensor, metagraph.
    """

    checkpoint: Any = None  # CheckpointWindow
    deltas: list[Any] = field(default_factory=list)  # list[DeltaWindow]
    accumulator_state: dict[str, Any] = field(default_factory=dict)
    wallet: Any = None
    subtensor: Any = None
    metagraph: Any = None
    config: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TaskHandler(Protocol):
    """Interface for auditor plugins."""

    name: str
    version: str

    async def on_cycle(self, context: AuditorContext) -> TaskResult:
        """Called each auditor cycle with the latest data."""
        ...


class PluginRegistry:
    """Discovers and dispatches to registered TaskHandler plugins."""

    def __init__(self) -> None:
        self._handlers: dict[str, TaskHandler] = {}

    def register(self, handler: TaskHandler) -> None:
        """Register a plugin handler."""
        if handler.name in self._handlers:
            raise ValueError(f"Plugin already registered: {handler.name}")
        self._handlers[handler.name] = handler
        bt.logging.info({"plugin_registered": handler.name, "version": handler.version})

    def discover(self, package: str) -> None:
        """Auto-discover and register plugins from a package.

        Imports all modules in the package and registers any module-level
        HANDLER attribute that implements TaskHandler.
        """
        try:
            pkg = importlib.import_module(package)
        except ImportError as e:
            bt.logging.warning({"plugin_discover_error": str(e)})
            return

        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            return

        for importer, modname, ispkg in pkgutil.iter_modules(pkg_path):
            if ispkg or modname.startswith("_"):
                continue
            try:
                mod = importlib.import_module(f"{package}.{modname}")
                handler = getattr(mod, "HANDLER", None)
                if handler is not None and isinstance(handler, TaskHandler):
                    self.register(handler)
            except Exception as e:
                bt.logging.warning({"plugin_import_error": {"module": modname, "error": str(e)}})

    @property
    def handlers(self) -> list[str]:
        """List registered handler names."""
        return list(self._handlers.keys())

    async def dispatch(self, context: AuditorContext) -> list[TaskResult]:
        """Run all registered plugins and collect results.

        Plugin failures are isolated - one crashing plugin doesn't
        prevent others from running.
        """
        results: list[TaskResult] = []
        for name, handler in self._handlers.items():
            try:
                result = await handler.on_cycle(context)
                results.append(result)
            except Exception as e:
                bt.logging.warning({"plugin_error": {"plugin": name, "error": str(e)}})
                results.append(TaskResult(
                    plugin_name=name,
                    plugin_version=getattr(handler, "version", "unknown"),
                    status="error",
                    evidence={"error": str(e)},
                ))
        return results


__all__ = ["AuditorContext", "PluginRegistry", "TaskHandler", "TaskResult"]
