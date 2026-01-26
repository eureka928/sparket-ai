"""Odds engines for fetching or generating market odds."""

from sparket.miner.base.engines.interface import OddsEngine, OddsPrices
from sparket.miner.base.engines.naive import NaiveEngine

__all__ = ["OddsEngine", "OddsPrices", "NaiveEngine"]

# Optional: TheOddsEngine requires API key
try:
    from sparket.miner.base.engines.theodds import TheOddsEngine
    __all__.append("TheOddsEngine")
except ImportError:
    pass








