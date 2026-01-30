"""Custom miner implementation optimized for scoring system.

This miner targets:
- 50% EconDim: Beat closing lines (CLV > 0)
- 30% InfoDim: Originality + lead market
- 20% Outcome accuracy (ForecastDim + SkillDim)
"""

from sparket.miner.custom.config import CustomMinerConfig
from sparket.miner.custom.runner import CustomMiner

__all__ = ["CustomMiner", "CustomMinerConfig"]
