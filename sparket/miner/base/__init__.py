"""
Base Miner - A reference implementation for Sparket miners.

This module provides a complete, self-contained miner implementation that:
- Fetches team statistics from ESPN (free, no API key required)
- Calculates team strength ratings using statistical models
- Generates odds using Log5 matchup probability
- Optionally enhances with The-Odds-API market data (free tier: 500 req/month)

Usage:
    from sparket.miner.base import BaseMiner
    
    base_miner = BaseMiner(
        hotkey=wallet.hotkey.ss58_address,
        config=config,
        validator_client=client,
        game_sync=sync,
    )
    await base_miner.start()

Miners can:
1. Use as-is: Works out of the box with ESPN data
2. Extend: Add custom stats providers for richer data
3. Replace: Swap out components or write entirely custom logic
"""

from sparket.miner.base.runner import BaseMiner
from sparket.miner.base.config import BaseMinerConfig

__all__ = ["BaseMiner", "BaseMinerConfig"]








