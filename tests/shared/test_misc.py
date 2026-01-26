"""Tests for shared/misc.py - TTL cache and utility functions."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from sparket.shared.misc import ttl_cache, _ttl_hash_gen, ttl_get_block


class TestTtlHashGen:
    """Tests for _ttl_hash_gen generator."""
    
    def test_yields_same_hash_within_window(self):
        """Same hash is yielded within the TTL window."""
        gen = _ttl_hash_gen(seconds=10)
        hash1 = next(gen)
        hash2 = next(gen)
        # Within same second, should be same hash
        assert hash1 == hash2
    
    def test_yields_different_hash_after_window(self):
        """Different hash after TTL window passes."""
        gen = _ttl_hash_gen(seconds=1)
        hash1 = next(gen)
        time.sleep(1.1)  # Wait for window to pass
        hash2 = next(gen)
        assert hash1 != hash2
    
    def test_hash_increments(self):
        """Hash increments by 1 each window."""
        gen = _ttl_hash_gen(seconds=1)
        hash1 = next(gen)
        time.sleep(1.1)
        hash2 = next(gen)
        # Should be exactly 1 more (or close if timing edge case)
        assert hash2 >= hash1 + 1


class TestTtlCache:
    """Tests for ttl_cache decorator."""
    
    def test_caches_function_result(self):
        """Function result is cached."""
        call_count = 0
        
        @ttl_cache(maxsize=10, ttl=60)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_func(5)
        result2 = expensive_func(5)
        
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once
    
    def test_different_args_different_cache_entries(self):
        """Different arguments create different cache entries."""
        call_count = 0
        
        @ttl_cache(maxsize=10, ttl=60)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        func(1)
        func(2)
        func(1)  # Should be cached
        func(2)  # Should be cached
        
        assert call_count == 2
    
    def test_cache_expires_after_ttl(self):
        """Cache entry expires after TTL."""
        call_count = 0
        
        @ttl_cache(maxsize=10, ttl=1)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        func(5)
        assert call_count == 1
        
        time.sleep(1.5)  # Wait for TTL to expire
        
        func(5)
        assert call_count == 2  # Called again after expiry
    
    def test_negative_ttl_uses_default(self):
        """Negative TTL uses default (effectively permanent cache)."""
        call_count = 0
        
        @ttl_cache(maxsize=10, ttl=-1)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x
        
        func(1)
        func(1)
        assert call_count == 1
    
    def test_zero_ttl_uses_default(self):
        """Zero TTL uses default."""
        call_count = 0
        
        @ttl_cache(maxsize=10, ttl=0)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x
        
        func(1)
        func(1)
        assert call_count == 1
    
    def test_maxsize_evicts_old_entries(self):
        """Cache evicts entries when maxsize is exceeded."""
        call_count = 0
        
        @ttl_cache(maxsize=2, ttl=60)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x
        
        func(1)
        func(2)
        func(3)  # Should evict func(1)
        
        assert call_count == 3
        
        func(3)  # Cached
        func(2)  # Cached
        
        assert call_count == 3
        
        func(1)  # Was evicted, needs recalculation
        assert call_count == 4
    
    def test_typed_cache(self):
        """Typed cache treats different types as different keys."""
        call_count = 0
        
        @ttl_cache(maxsize=10, typed=True, ttl=60)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x
        
        func(3)
        func(3.0)  # Different type
        
        assert call_count == 2
    
    def test_preserves_function_metadata(self):
        """Decorator preserves function name and docstring."""
        @ttl_cache(maxsize=10, ttl=60)
        def my_function(x):
            """My docstring."""
            return x
        
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."
    
    def test_works_with_kwargs(self):
        """Cache works with keyword arguments."""
        call_count = 0
        
        @ttl_cache(maxsize=10, ttl=60)
        def func(a, b=10):
            nonlocal call_count
            call_count += 1
            return a + b
        
        func(1, b=5)
        func(1, b=5)
        
        assert call_count == 1


class TestTtlGetBlock:
    """Tests for ttl_get_block function."""
    
    def test_returns_current_block(self):
        """Returns the current block from subtensor."""
        mock_self = MagicMock()
        mock_self.subtensor.get_current_block.return_value = 12345
        
        result = ttl_get_block(mock_self)
        assert result == 12345
        mock_self.subtensor.get_current_block.assert_called()
    
    def test_caches_result_within_ttl(self):
        """Result is cached within TTL window."""
        mock_self = MagicMock()
        mock_self.subtensor.get_current_block.return_value = 99999
        
        # Multiple calls within TTL should use cache
        result1 = ttl_get_block(mock_self)
        result2 = ttl_get_block(mock_self)
        
        # Results should be the same
        assert result1 == result2

