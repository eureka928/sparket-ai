"""Data fetchers for scores, standings, and team statistics."""

from sparket.miner.base.fetchers.interface import ScoreFetcher, StatsFetcher, GameResult
from sparket.miner.base.fetchers.espn import ESPNFetcher

__all__ = ["ScoreFetcher", "StatsFetcher", "GameResult", "ESPNFetcher"]








