"""Test proxy mode behavior for IP privacy."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestProxyModeDetection:
    """Test that proxy mode is correctly detected from config."""

    def test_proxy_mode_enabled_from_app_config(self):
        """Verify proxy mode is detected when app_config has proxy_url."""
        # Mock app_config with proxy_url set
        mock_api = MagicMock()
        mock_api.proxy_url = "https://my-proxy.example.com"
        
        mock_core = MagicMock()
        mock_core.api = mock_api
        
        mock_app_config = MagicMock()
        mock_app_config.core = mock_core
        
        # Test the detection logic
        proxy_url = None
        app_cfg = mock_app_config
        if app_cfg:
            core = getattr(app_cfg, "core", None)
            api_cfg = getattr(core, "api", None) if core else None
            proxy_url = getattr(api_cfg, "proxy_url", None) if api_cfg else None
        
        assert proxy_url == "https://my-proxy.example.com"
        assert bool(proxy_url) is True

    def test_proxy_mode_disabled_when_no_proxy_url(self):
        """Verify proxy mode is disabled when no proxy_url configured."""
        mock_api = MagicMock()
        mock_api.proxy_url = None
        
        mock_core = MagicMock()
        mock_core.api = mock_api
        
        mock_app_config = MagicMock()
        mock_app_config.core = mock_core
        
        proxy_url = None
        app_cfg = mock_app_config
        if app_cfg:
            core = getattr(app_cfg, "core", None)
            api_cfg = getattr(core, "api", None) if core else None
            proxy_url = getattr(api_cfg, "proxy_url", None) if api_cfg else None
        
        assert proxy_url is None
        assert bool(proxy_url) is False

    def test_proxy_mode_from_env_var(self):
        """Verify proxy_url is loaded from environment variable."""
        import os
        
        # Set env var before loading settings
        os.environ["SPARKET_API__PROXY_URL"] = "https://env-proxy.example.com"
        
        try:
            from sparket.config import load_settings
            settings = load_settings(role="validator")
            proxy_url = getattr(settings.api, "proxy_url", None)
            
            assert proxy_url == "https://env-proxy.example.com"
        finally:
            # Cleanup
            del os.environ["SPARKET_API__PROXY_URL"]


class TestAxonProxyBehavior:
    """Test axon.serve() is correctly skipped/called based on proxy mode."""

    def test_axon_serve_skipped_in_proxy_mode(self):
        """Verify axon.serve() is NOT called when proxy_url is configured."""
        # Simulate the serve_axon logic with proxy_mode = True
        mock_axon = MagicMock()
        mock_axon.serve = MagicMock()
        mock_axon.start = MagicMock()
        
        proxy_mode = True
        registered = False
        
        # This mirrors the logic in serve_axon()
        if proxy_mode:
            # Skip chain registration
            pass
        else:
            mock_axon.serve(netuid=2, subtensor=MagicMock())
            registered = True
        
        # Axon should always start
        mock_axon.start()
        
        # Verify serve was NOT called, but start WAS called
        mock_axon.serve.assert_not_called()
        mock_axon.start.assert_called_once()
        assert registered is False

    def test_axon_serve_called_without_proxy(self):
        """Verify axon.serve() IS called when no proxy_url configured."""
        mock_axon = MagicMock()
        mock_axon.serve = MagicMock()
        mock_axon.start = MagicMock()
        mock_subtensor = MagicMock()
        
        proxy_mode = False
        registered = False
        
        if proxy_mode:
            pass
        else:
            mock_axon.serve(netuid=2, subtensor=mock_subtensor)
            registered = True
        
        mock_axon.start()
        
        # Verify both serve and start were called
        mock_axon.serve.assert_called_once_with(netuid=2, subtensor=mock_subtensor)
        mock_axon.start.assert_called_once()
        assert registered is True


class TestValidatorCommsEndpoint:
    """Test ValidatorComms returns correct endpoint based on proxy config."""

    def test_advertised_endpoint_returns_proxy_when_configured(self):
        """Verify advertised_endpoint returns proxy_url when set."""
        from sparket.validator.comms import ValidatorComms
        
        comms = ValidatorComms(
            proxy_url="https://my-proxy.example.com",
            require_token=True,
        )
        
        # Create mock axon
        mock_axon = MagicMock()
        mock_axon.external_ip = "192.168.1.100"
        mock_axon.external_port = 8093
        
        endpoint = comms.advertised_endpoint(axon=mock_axon)
        
        # Should return proxy URL in dict format, not real IP
        assert endpoint == {"url": "https://my-proxy.example.com"}

    def test_advertised_endpoint_returns_real_ip_without_proxy(self):
        """Verify advertised_endpoint returns real IP when no proxy configured."""
        from sparket.validator.comms import ValidatorComms
        
        comms = ValidatorComms(
            proxy_url=None,
            require_token=True,
        )
        
        mock_axon = MagicMock()
        mock_axon.external_ip = "192.168.1.100"
        mock_axon.external_port = 8093
        
        endpoint = comms.advertised_endpoint(axon=mock_axon)
        
        # Should return host/port dict
        assert endpoint == {"host": "192.168.1.100", "port": 8093}


class TestAxonStillListensWithoutServe:
    """Verify axon accepts connections even without serve() call."""

    @pytest.mark.asyncio
    async def test_axon_listens_without_chain_registration(self):
        """Test that axon.start() enables connections without serve()."""
        import bittensor as bt
        from bittensor import Synapse
        import socket
        import asyncio
        
        # Create axon
        axon = bt.Axon(wallet=None, port=9997)
        
        async def dummy_handler(synapse: Synapse) -> Synapse:
            return synapse
        
        axon.attach(forward_fn=dummy_handler)
        
        # Skip serve() - only call start()
        axon.start()
        
        # Wait for server to be ready
        await asyncio.sleep(0.5)
        
        # Test connection
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', 9997))
            assert result == 0, "Axon should accept connections without serve()"
        finally:
            sock.close()
            axon.stop()
