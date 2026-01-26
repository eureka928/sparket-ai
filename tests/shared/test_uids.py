"""Tests for shared/uids.py - UID availability and random selection utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from sparket.shared.uids import check_uid_availability, get_random_uids


class MockAxon:
    """Mock axon for testing."""
    def __init__(self, is_serving: bool = True):
        self.is_serving = is_serving


class MockMetagraph:
    """Mock metagraph for testing UID functions."""
    def __init__(
        self,
        n_neurons: int = 10,
        serving_mask: list = None,
        validator_permit_mask: list = None,
        stakes: list = None,
    ):
        self.n = MagicMock()
        self.n.item.return_value = n_neurons
        
        # Default all serving
        if serving_mask is None:
            serving_mask = [True] * n_neurons
        self.axons = [MockAxon(is_serving=s) for s in serving_mask]
        
        # Default no validator permits
        if validator_permit_mask is None:
            validator_permit_mask = [False] * n_neurons
        self.validator_permit = validator_permit_mask
        
        # Default low stakes
        if stakes is None:
            stakes = [100.0] * n_neurons
        self.S = stakes


class TestCheckUidAvailability:
    """Tests for check_uid_availability function."""
    
    def test_available_when_serving_no_permit(self):
        """UID is available when serving and has no validator permit."""
        metagraph = MockMetagraph(n_neurons=5)
        assert check_uid_availability(metagraph, uid=0, vpermit_tao_limit=1024) is True
    
    def test_not_available_when_not_serving(self):
        """UID is not available when not serving."""
        metagraph = MockMetagraph(
            n_neurons=5,
            serving_mask=[True, False, True, True, True],
        )
        assert check_uid_availability(metagraph, uid=1, vpermit_tao_limit=1024) is False
    
    def test_available_with_permit_under_limit(self):
        """UID with validator permit is available if stake under limit."""
        metagraph = MockMetagraph(
            n_neurons=5,
            validator_permit_mask=[True, False, False, False, False],
            stakes=[500.0, 100.0, 100.0, 100.0, 100.0],
        )
        assert check_uid_availability(metagraph, uid=0, vpermit_tao_limit=1024) is True
    
    def test_not_available_with_permit_over_limit(self):
        """UID with validator permit is NOT available if stake over limit."""
        metagraph = MockMetagraph(
            n_neurons=5,
            validator_permit_mask=[True, False, False, False, False],
            stakes=[2000.0, 100.0, 100.0, 100.0, 100.0],
        )
        assert check_uid_availability(metagraph, uid=0, vpermit_tao_limit=1024) is False
    
    def test_available_with_permit_at_exact_limit(self):
        """UID with validator permit at exact limit is available (not over)."""
        metagraph = MockMetagraph(
            n_neurons=5,
            validator_permit_mask=[True, False, False, False, False],
            stakes=[1024.0, 100.0, 100.0, 100.0, 100.0],
        )
        # At exactly 1024, not OVER, so should be available
        assert check_uid_availability(metagraph, uid=0, vpermit_tao_limit=1024) is True
    
    def test_edge_case_zero_stake(self):
        """UID with zero stake is available."""
        metagraph = MockMetagraph(
            n_neurons=3,
            stakes=[0.0, 0.0, 0.0],
        )
        assert check_uid_availability(metagraph, uid=0, vpermit_tao_limit=1024) is True


class TestGetRandomUids:
    """Tests for get_random_uids function."""
    
    def setup_method(self):
        """Create a mock self object with metagraph and config."""
        self.mock_self = MagicMock()
        self.mock_self.metagraph = MockMetagraph(n_neurons=10)
        self.mock_self.config.neuron.vpermit_tao_limit = 1024
    
    def test_returns_requested_count(self):
        """Returns k UIDs when k available UIDs exist."""
        result = get_random_uids(self.mock_self, k=5)
        assert len(result) == 5
        assert isinstance(result, np.ndarray)
    
    def test_returns_fewer_when_not_enough_available(self):
        """Returns fewer than k when not enough UIDs available."""
        # Only 3 serving
        self.mock_self.metagraph = MockMetagraph(
            n_neurons=10,
            serving_mask=[True, True, True, False, False, False, False, False, False, False],
        )
        result = get_random_uids(self.mock_self, k=5)
        assert len(result) == 3
    
    def test_excludes_specified_uids(self):
        """Respects exclude list when sufficient alternatives exist."""
        result = get_random_uids(self.mock_self, k=3, exclude=[0, 1, 2])
        # Should not contain excluded UIDs (if enough alternatives)
        for uid in result:
            assert uid not in [0, 1, 2]
    
    def test_uses_excluded_when_necessary(self):
        """Uses excluded UIDs when not enough non-excluded available."""
        # Only 3 UIDs available, exclude 2 of them
        self.mock_self.metagraph = MockMetagraph(
            n_neurons=10,
            serving_mask=[True, True, True, False, False, False, False, False, False, False],
        )
        result = get_random_uids(self.mock_self, k=3, exclude=[0, 1])
        # Must include some excluded UIDs to meet k=3
        assert len(result) == 3
    
    def test_returns_unique_uids(self):
        """All returned UIDs are unique."""
        result = get_random_uids(self.mock_self, k=5)
        assert len(result) == len(set(result))
    
    def test_empty_exclude_list(self):
        """Works with empty exclude list."""
        result = get_random_uids(self.mock_self, k=3, exclude=[])
        assert len(result) == 3
    
    def test_none_exclude_list(self):
        """Works with None exclude list."""
        result = get_random_uids(self.mock_self, k=3, exclude=None)
        assert len(result) == 3
    
    def test_k_zero(self):
        """Returns empty array when k=0."""
        result = get_random_uids(self.mock_self, k=0)
        assert len(result) == 0
    
    def test_no_available_uids(self):
        """Handles case where no UIDs are available."""
        self.mock_self.metagraph = MockMetagraph(
            n_neurons=5,
            serving_mask=[False, False, False, False, False],
        )
        result = get_random_uids(self.mock_self, k=3)
        assert len(result) == 0
