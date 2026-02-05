"""Tests for the auditor plugin registry."""

import pytest
from datetime import datetime, timezone

from sparket.validator.auditor.plugin_registry import (
    AuditorContext,
    PluginRegistry,
    TaskHandler,
    TaskResult,
)


class MockHandler:
    """Simple mock plugin handler."""

    def __init__(self, name: str = "mock", status: str = "pass"):
        self.name = name
        self.version = "1.0.0"
        self._status = status
        self.call_count = 0

    async def on_cycle(self, context: AuditorContext) -> TaskResult:
        self.call_count += 1
        return TaskResult(
            plugin_name=self.name,
            plugin_version=self.version,
            status=self._status,
            evidence={"called": True},
            completed_at=datetime.now(timezone.utc),
        )


class FailingHandler:
    name = "failing"
    version = "1.0.0"

    async def on_cycle(self, context: AuditorContext) -> TaskResult:
        raise RuntimeError("Plugin crashed!")


class TestPluginRegistry:

    def test_register_and_discover(self):
        registry = PluginRegistry()
        handler = MockHandler("test_plugin")
        registry.register(handler)
        assert "test_plugin" in registry.handlers

    @pytest.mark.asyncio
    async def test_dispatch_calls_all_handlers(self):
        registry = PluginRegistry()
        h1 = MockHandler("plugin_a")
        h2 = MockHandler("plugin_b")
        registry.register(h1)
        registry.register(h2)

        context = AuditorContext()
        results = await registry.dispatch(context)

        assert len(results) == 2
        assert h1.call_count == 1
        assert h2.call_count == 1

    @pytest.mark.asyncio
    async def test_dispatch_isolates_failures(self):
        """One plugin crashing doesn't prevent others from running."""
        registry = PluginRegistry()
        good = MockHandler("good")
        bad = FailingHandler()
        registry.register(good)
        registry.register(bad)

        context = AuditorContext()
        results = await registry.dispatch(context)

        assert len(results) == 2
        statuses = {r.plugin_name: r.status for r in results}
        assert statuses["good"] == "pass"
        assert statuses["failing"] == "error"
        assert good.call_count == 1

    @pytest.mark.asyncio
    async def test_handler_results_collected(self):
        registry = PluginRegistry()
        registry.register(MockHandler("a", status="pass"))
        registry.register(MockHandler("b", status="fail"))

        results = await registry.dispatch(AuditorContext())
        statuses = {r.plugin_name: r.status for r in results}
        assert statuses["a"] == "pass"
        assert statuses["b"] == "fail"

    def test_duplicate_registration_rejected(self):
        registry = PluginRegistry()
        registry.register(MockHandler("dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockHandler("dup"))

    @pytest.mark.asyncio
    async def test_empty_registry_dispatch_succeeds(self):
        registry = PluginRegistry()
        results = await registry.dispatch(AuditorContext())
        assert results == []

    def test_handlers_property(self):
        registry = PluginRegistry()
        registry.register(MockHandler("x"))
        registry.register(MockHandler("y"))
        assert sorted(registry.handlers) == ["x", "y"]
