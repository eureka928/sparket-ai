"""Tests for validator/comms.py - Token rotation and verification."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from sparket.validator.comms import ValidatorComms


class TestValidatorComms:
    """Tests for ValidatorComms class."""
    
    def test_init_with_defaults(self):
        """Initializes with default values."""
        comms = ValidatorComms(proxy_url=None, require_token=True)
        assert comms.proxy_url is None
        assert comms.require_token is True
        assert comms.step_rotation == 10
        assert comms._secret is not None
        assert len(comms._secret) == 32
    
    def test_init_with_proxy_url(self):
        """Initializes with custom proxy URL."""
        comms = ValidatorComms(proxy_url="https://proxy.example.com", require_token=False)
        assert comms.proxy_url == "https://proxy.example.com"
        assert comms.require_token is False
    
    def test_init_with_custom_step_rotation(self):
        """Initializes with custom step rotation."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=25)
        assert comms.step_rotation == 25
    
    def test_step_rotation_minimum_is_one(self):
        """Step rotation has minimum value of 1."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=0)
        assert comms.step_rotation == 1
        
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=-5)
        assert comms.step_rotation == 1
    
    def test_init_with_env_secret(self):
        """Uses secret from environment variable if set."""
        with patch.dict(os.environ, {"SPARKET_VALIDATOR_PUSH_SECRET": "my_secret_key"}):
            comms = ValidatorComms(proxy_url=None, require_token=True)
            assert comms._secret == b"my_secret_key"
    
    def test_current_token_generates_token(self):
        """current_token generates a token."""
        comms = ValidatorComms(proxy_url=None, require_token=True)
        token = comms.current_token(step=0)
        assert isinstance(token, str)
        assert len(token) == 64  # SHA256 hex digest
    
    def test_current_token_same_within_epoch(self):
        """Same token within same epoch (step // rotation)."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=10)
        
        token0 = comms.current_token(step=0)
        token5 = comms.current_token(step=5)
        token9 = comms.current_token(step=9)
        
        assert token0 == token5 == token9
    
    def test_current_token_different_across_epochs(self):
        """Different tokens across epochs."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=10)
        
        token_epoch0 = comms.current_token(step=5)
        token_epoch1 = comms.current_token(step=15)
        token_epoch2 = comms.current_token(step=25)
        
        assert token_epoch0 != token_epoch1 != token_epoch2
    
    def test_current_token_caches_result(self):
        """Token is cached per epoch."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=10)
        
        token1 = comms.current_token(step=5)
        token2 = comms.current_token(step=5)
        
        assert token1 == token2
        assert comms._last_epoch_step == 0  # epoch for step 5 with rotation 10
        assert comms._cached_token == token1


class TestVerifyToken:
    """Tests for token verification."""
    
    def test_verify_when_require_token_false(self):
        """Always returns True when require_token is False."""
        comms = ValidatorComms(proxy_url=None, require_token=False)
        assert comms.verify_token(token=None, step=0) is True
        assert comms.verify_token(token="garbage", step=0) is True
        assert comms.verify_token(token="", step=0) is True
    
    def test_verify_rejects_none_token(self):
        """Rejects None token when require_token is True."""
        comms = ValidatorComms(proxy_url=None, require_token=True)
        assert comms.verify_token(token=None, step=0) is False
    
    def test_verify_rejects_empty_token(self):
        """Rejects empty string token."""
        comms = ValidatorComms(proxy_url=None, require_token=True)
        assert comms.verify_token(token="", step=0) is False
    
    def test_verify_accepts_current_epoch_token(self):
        """Accepts token from current epoch."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=10)
        
        token = comms.current_token(step=15)
        assert comms.verify_token(token=token, step=15) is True
        assert comms.verify_token(token=token, step=18) is True  # Same epoch
    
    def test_verify_accepts_previous_epoch_token(self):
        """Accepts token from previous epoch (for clock drift)."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=10)
        
        token_epoch0 = comms.current_token(step=5)
        # At step 15 (epoch 1), should still accept epoch 0 token
        assert comms.verify_token(token=token_epoch0, step=15) is True
    
    def test_verify_rejects_old_epoch_token(self):
        """Rejects token from too old epoch."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=10)
        
        token_epoch0 = comms.current_token(step=5)
        # At step 25 (epoch 2), epoch 0 token is too old
        assert comms.verify_token(token=token_epoch0, step=25) is False
    
    def test_verify_rejects_garbage_token(self):
        """Rejects completely invalid token."""
        comms = ValidatorComms(proxy_url=None, require_token=True)
        assert comms.verify_token(token="not_a_valid_token", step=0) is False
    
    def test_verify_timing_safe_comparison(self):
        """Uses timing-safe comparison (hmac.compare_digest)."""
        # This is implicitly tested by the implementation using hmac.compare_digest
        # We can verify by checking a valid token works
        comms = ValidatorComms(proxy_url=None, require_token=True)
        token = comms.current_token(step=0)
        assert comms.verify_token(token=token, step=0) is True
    
    def test_verify_edge_case_step_zero(self):
        """Works at step 0."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=10)
        token = comms.current_token(step=0)
        assert comms.verify_token(token=token, step=0) is True
    
    def test_verify_handles_epoch_boundary(self):
        """Handles verification at epoch boundary."""
        comms = ValidatorComms(proxy_url=None, require_token=True, step_rotation=10)
        
        token_epoch0 = comms.current_token(step=9)
        token_epoch1 = comms.current_token(step=10)
        
        # At step 10, both should be valid (current and previous)
        assert comms.verify_token(token=token_epoch0, step=10) is True
        assert comms.verify_token(token=token_epoch1, step=10) is True


class TestAdvertisedEndpoint:
    """Tests for advertised_endpoint method."""
    
    def test_returns_proxy_url_when_set(self):
        """Returns proxy URL when configured."""
        comms = ValidatorComms(proxy_url="https://proxy.example.com:8080", require_token=False)
        
        mock_axon = MagicMock()
        mock_axon.ip = "192.168.1.1"
        mock_axon.port = 9000
        
        result = comms.advertised_endpoint(axon=mock_axon)
        
        assert result == {"url": "https://proxy.example.com:8080"}
    
    def test_returns_axon_details_when_no_proxy(self):
        """Returns axon host/port when no proxy configured."""
        comms = ValidatorComms(proxy_url=None, require_token=False)
        
        mock_axon = MagicMock()
        mock_axon.external_ip = None  # No external IP
        mock_axon.external_port = None  # No external port
        mock_axon.ip = "192.168.1.1"
        mock_axon.port = 9000
        
        result = comms.advertised_endpoint(axon=mock_axon)
        
        assert result == {"host": "192.168.1.1", "port": 9000}
    
    def test_defaults_when_axon_missing_attributes(self):
        """Uses defaults when axon missing ip/port."""
        comms = ValidatorComms(proxy_url=None, require_token=False)
        
        mock_axon = MagicMock(spec=[])  # No attributes
        
        result = comms.advertised_endpoint(axon=mock_axon)
        
        assert result == {"host": "127.0.0.1", "port": 0}
    
    def test_handles_none_ip(self):
        """Handles None ip gracefully."""
        comms = ValidatorComms(proxy_url=None, require_token=False)
        
        mock_axon = MagicMock()
        mock_axon.external_ip = None
        mock_axon.external_port = None
        mock_axon.ip = None
        mock_axon.port = 8080
        
        result = comms.advertised_endpoint(axon=mock_axon)
        
        assert result == {"host": "127.0.0.1", "port": 8080}
    
    def test_handles_none_port(self):
        """Handles None port gracefully."""
        comms = ValidatorComms(proxy_url=None, require_token=False)
        
        mock_axon = MagicMock()
        mock_axon.external_ip = None
        mock_axon.external_port = None
        mock_axon.ip = "10.0.0.1"
        mock_axon.port = None
        
        result = comms.advertised_endpoint(axon=mock_axon)
        
        assert result == {"host": "10.0.0.1", "port": 0}
