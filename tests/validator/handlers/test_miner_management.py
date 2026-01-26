"""Tests for miner registration/deregistration handling."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from sparket.validator.handlers.core.chain.miner_management import (
    MinerManagementHandler,
)


class MockMetagraph:
    """Mock metagraph for testing."""
    
    def __init__(self, hotkeys: list, coldkeys: list = None):
        self.hotkeys = hotkeys
        self.coldkeys = coldkeys or hotkeys
        self.S = [0.0] * len(hotkeys)  # stake
        self.R = [0.0] * len(hotkeys)  # rank
        self.E = [0.0] * len(hotkeys)  # emission
        self.I = [0.0] * len(hotkeys)  # incentive
        self.C = [0.0] * len(hotkeys)  # consensus
        self.T = [0.0] * len(hotkeys)  # trust
        self.Tv = [0.0] * len(hotkeys)  # validator trust
        self.D = [0.0] * len(hotkeys)  # dividends
        self.validator_permit = [False] * len(hotkeys)


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.miners = {}  # hotkey -> miner data
        self.miners_by_id = {}  # miner_id -> hotkey (for delete lookups)
        self.writes = []
        self.reads = []
        self._next_id = 1
    
    async def read(self, query, params=None, mappings=False):
        self.reads.append({"query": str(query), "params": params})
        
        query_str = str(query)
        
        if "FROM miner" in query_str and "active = 1" in query_str:
            # Return miners for the netuid (all are "active" in mock)
            result = [
                {"miner_id": m["miner_id"], "hotkey": hk, "uid": m["uid"], "netuid": m["netuid"]}
                for hk, m in self.miners.items()
                if m.get("netuid") == params.get("netuid")
            ]
            return result
        
        return []
    
    async def write(self, query, params=None, *, return_rows=False, mappings=False):
        self.writes.append({"query": str(query), "params": params})
        
        query_str = str(query)
        
        # Handle INSERT INTO miner (upsert) - this is a write with RETURNING
        if "INSERT INTO miner" in query_str:
            hotkey = params.get("hotkey")
            if hotkey in self.miners:
                # Update existing
                self.miners[hotkey].update({
                    "uid": params.get("uid"),
                    "netuid": params.get("netuid"),
                })
                if return_rows:
                    return [{"miner_id": self.miners[hotkey]["miner_id"], "was_inserted": False}]
                return 1
            else:
                # Insert new
                miner_id = self._next_id
                self._next_id += 1
                self.miners[hotkey] = {
                    "miner_id": miner_id,
                    "hotkey": hotkey,
                    "coldkey": params.get("coldkey"),
                    "uid": params.get("uid"),
                    "netuid": params.get("netuid"),
                }
                self.miners_by_id[miner_id] = hotkey
                if return_rows:
                    return [{"miner_id": miner_id, "was_inserted": True}]
                return 1
        
        # Handle DELETE FROM miner
        if "DELETE FROM miner WHERE miner_id" in query_str:
            miner_id = params.get("miner_id")
            if miner_id in self.miners_by_id:
                hotkey = self.miners_by_id[miner_id]
                del self.miners[hotkey]
                del self.miners_by_id[miner_id]
        
        return 1


class TestMinerManagementHandler:
    """Test suite for MinerManagementHandler."""
    
    @pytest.mark.asyncio
    async def test_sync_new_miners(self):
        """Test that new miners are inserted into database."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        metagraph = MockMetagraph(
            hotkeys=["hotkey_0", "hotkey_1", "hotkey_2"]
        )
        
        result = await handler.sync_metagraph_to_db(
            metagraph=metagraph,
            block=1000,
            netuid=2,
        )
        
        assert result["inserted"] == 3
        assert result["updated"] == 0
        assert result["deleted"] == 0
        assert len(db.miners) == 3
    
    @pytest.mark.asyncio
    async def test_detect_hotkey_change_deletes_old_miner(self):
        """Test that hotkey changes delete the old miner entirely."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        # First sync - establish initial state
        metagraph1 = MockMetagraph(
            hotkeys=["hotkey_0", "hotkey_1", "hotkey_2"]
        )
        await handler.sync_metagraph_to_db(metagraph1, block=1000, netuid=2)
        assert len(db.miners) == 3
        
        # Second sync - hotkey_1 replaced by new_hotkey_1
        metagraph2 = MockMetagraph(
            hotkeys=["hotkey_0", "new_hotkey_1", "hotkey_2"]
        )
        result = await handler.sync_metagraph_to_db(metagraph2, block=2000, netuid=2)
        
        # Should detect the deregistration and DELETE old miner
        assert result["deleted"] == 1
        assert len(result["hotkey_changes"]) == 1
        
        # Old miner should be GONE (deleted, not just inactive)
        assert "hotkey_1" not in db.miners
        
        # New miner should exist
        assert "new_hotkey_1" in db.miners
        assert db.miners["new_hotkey_1"]["uid"] == 1
    
    @pytest.mark.asyncio
    async def test_old_miner_data_deleted_on_replacement(self):
        """Test that old miner's data is deleted (not preserved) after deregistration."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        # Establish initial state
        metagraph1 = MockMetagraph(hotkeys=["old_miner"])
        await handler.sync_metagraph_to_db(metagraph1, block=1000, netuid=2)
        
        assert "old_miner" in db.miners
        old_miner_id = db.miners["old_miner"]["miner_id"]
        
        # New miner takes over
        metagraph2 = MockMetagraph(hotkeys=["new_miner"])
        await handler.sync_metagraph_to_db(metagraph2, block=2000, netuid=2)
        
        # Old miner record should be DELETED
        assert "old_miner" not in db.miners
        assert old_miner_id not in db.miners_by_id
        
        # New miner should exist
        assert "new_miner" in db.miners
    
    def test_check_miners_registered_detects_changes(self):
        """Test change detection between old and new hotkey lists."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        old_hotkeys = ["hk0", "hk1", "hk2", "hk3"]
        metagraph = MockMetagraph(hotkeys=["hk0", "NEW_hk1", "hk2", "NEW_hk3"])
        
        changes = handler.check_miners_registered(metagraph, old_hotkeys)
        
        assert len(changes) == 2
        assert (1, "hk1", "NEW_hk1") in changes
        assert (3, "hk3", "NEW_hk3") in changes
    
    @pytest.mark.asyncio
    async def test_multiple_deregistrations_same_sync(self):
        """Test handling multiple deregistrations in single sync."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        # Initial state with 5 miners
        metagraph1 = MockMetagraph(
            hotkeys=["m0", "m1", "m2", "m3", "m4"]
        )
        await handler.sync_metagraph_to_db(metagraph1, block=1000, netuid=2)
        assert len(db.miners) == 5
        
        # 3 miners replaced at once
        metagraph2 = MockMetagraph(
            hotkeys=["m0", "new1", "new2", "new3", "m4"]
        )
        result = await handler.sync_metagraph_to_db(metagraph2, block=2000, netuid=2)
        
        assert result["deleted"] == 3
        assert len(result["hotkey_changes"]) == 3
        
        # Old miners should be DELETED
        assert "m1" not in db.miners
        assert "m2" not in db.miners
        assert "m3" not in db.miners
        
        # New miners should exist
        assert "new1" in db.miners
        assert "new2" in db.miners
        assert "new3" in db.miners
        
        # Unchanged miners still exist
        assert "m0" in db.miners
        assert "m4" in db.miners


class TestSyncMetagraphHandler:
    """Test the full sync flow including metagraph handler."""
    
    def test_should_sync(self):
        """Test sync timing logic."""
        from sparket.validator.handlers.core.chain.sync_metagraph import SyncMetagraphHandler
        
        db = MockDatabase()
        handler = SyncMetagraphHandler(db)
        
        # Create mock validator
        validator = MagicMock()
        validator.block = 1000
        validator.uid = 0
        validator.metagraph = MagicMock()
        validator.metagraph.last_update = [900]  # 100 blocks ago
        validator.config = MagicMock()
        validator.config.neuron = MagicMock()
        validator.config.neuron.epoch_length = 50  # Sync every 50 blocks
        
        assert handler.should_sync(validator) is True
        
        # If more recent update
        validator.metagraph.last_update = [980]  # Only 20 blocks ago
        assert handler.should_sync(validator) is False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_metagraph(self):
        """Test handling empty metagraph."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        metagraph = MockMetagraph(hotkeys=[])
        result = await handler.sync_metagraph_to_db(metagraph, block=1000, netuid=2)
        
        assert result["inserted"] == 0
        assert result["updated"] == 0
        assert result["deleted"] == 0
    
    @pytest.mark.asyncio
    async def test_empty_hotkey_skipped(self):
        """Test that empty hotkeys are skipped."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        metagraph = MockMetagraph(hotkeys=["valid_key", "", "another_key"])
        result = await handler.sync_metagraph_to_db(metagraph, block=1000, netuid=2)
        
        # Only 2 should be inserted (empty string skipped)
        assert result["inserted"] == 2
        assert "valid_key" in db.miners
        assert "" not in db.miners
        assert "another_key" in db.miners
    
    @pytest.mark.asyncio
    async def test_same_hotkey_different_sync_is_update(self):
        """Test that syncing same miner twice is an update, not insert."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        metagraph = MockMetagraph(hotkeys=["miner1"])
        
        # First sync
        result1 = await handler.sync_metagraph_to_db(metagraph, block=1000, netuid=2)
        assert result1["inserted"] == 1
        assert result1["updated"] == 0
        
        # Second sync with same hotkey
        result2 = await handler.sync_metagraph_to_db(metagraph, block=2000, netuid=2)
        assert result2["inserted"] == 0
        assert result2["updated"] == 1
    
    @pytest.mark.asyncio
    async def test_uid_reuse_after_deregistration(self):
        """Test that a UID can be reused after the old miner is deleted."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        # First miner at UID 0
        metagraph1 = MockMetagraph(hotkeys=["first_miner"])
        await handler.sync_metagraph_to_db(metagraph1, block=1000, netuid=2)
        
        first_miner_id = db.miners["first_miner"]["miner_id"]
        assert db.miners["first_miner"]["uid"] == 0
        
        # New miner takes UID 0
        metagraph2 = MockMetagraph(hotkeys=["second_miner"])
        await handler.sync_metagraph_to_db(metagraph2, block=2000, netuid=2)
        
        # First miner should be DELETED
        assert "first_miner" not in db.miners
        
        # Second miner should have uid=0
        assert db.miners["second_miner"]["uid"] == 0
        
        # Second miner has different miner_id
        assert db.miners["second_miner"]["miner_id"] != first_miner_id
    
    @pytest.mark.asyncio
    async def test_rapid_succession_replacements(self):
        """Test multiple rapid replacements at same UID."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        # First miner
        await handler.sync_metagraph_to_db(
            MockMetagraph(hotkeys=["miner_a"]), block=1000, netuid=2
        )
        assert "miner_a" in db.miners
        
        # Quick replacement - miner_a deleted, miner_b inserted
        await handler.sync_metagraph_to_db(
            MockMetagraph(hotkeys=["miner_b"]), block=1001, netuid=2
        )
        assert "miner_a" not in db.miners
        assert "miner_b" in db.miners
        
        # Another quick replacement - miner_b deleted, miner_c inserted
        await handler.sync_metagraph_to_db(
            MockMetagraph(hotkeys=["miner_c"]), block=1002, netuid=2
        )
        assert "miner_b" not in db.miners
        assert "miner_c" in db.miners
        
        # Only miner_c should exist
        assert len(db.miners) == 1
        assert db.miners["miner_c"]["uid"] == 0
    
    @pytest.mark.asyncio
    async def test_different_netuids_isolated(self):
        """Test that different netuids don't interfere."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        # Miners on netuid 1
        await handler.sync_metagraph_to_db(
            MockMetagraph(hotkeys=["net1_miner"]), block=1000, netuid=1
        )
        
        # Miners on netuid 2
        await handler.sync_metagraph_to_db(
            MockMetagraph(hotkeys=["net2_miner"]), block=1000, netuid=2
        )
        
        # Both should exist
        assert db.miners["net1_miner"]["netuid"] == 1
        assert db.miners["net2_miner"]["netuid"] == 2
    
    def test_check_miners_registered_empty_lists(self):
        """Test change detection with empty lists."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        changes = handler.check_miners_registered(MockMetagraph([]), [])
        assert changes == []
    
    def test_check_miners_registered_same_lists(self):
        """Test change detection when nothing changed."""
        db = MockDatabase()
        handler = MinerManagementHandler(db)
        
        hotkeys = ["hk0", "hk1", "hk2"]
        changes = handler.check_miners_registered(MockMetagraph(hotkeys), hotkeys)
        assert changes == []
