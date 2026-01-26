"""Tests for validator/handlers/handlers.py - Handler registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sparket.validator.handlers.handlers import Handlers
from sparket.validator.handlers.ingest.ingest_odds import IngestOddsHandler
from sparket.validator.handlers.ingest.ingest_outcome import IngestOutcomeHandler
from sparket.validator.handlers.data.game_data import GameDataHandler


class TestHandlers:
    """Tests for Handlers registry class."""
    
    def test_initializes_with_database(self):
        """Initializes all handlers with the provided database."""
        mock_db = MagicMock()
        
        handlers = Handlers(mock_db)
        
        assert handlers.database is mock_db
    
    def test_creates_ingest_odds_handler(self):
        """Creates IngestOddsHandler instance."""
        mock_db = MagicMock()
        handlers = Handlers(mock_db)
        
        assert isinstance(handlers.ingest_odds_handler, IngestOddsHandler)
        assert handlers.ingest_odds_handler.database is mock_db
    
    def test_creates_ingest_outcome_handler(self):
        """Creates IngestOutcomeHandler instance."""
        mock_db = MagicMock()
        handlers = Handlers(mock_db)
        
        assert isinstance(handlers.ingest_outcome_handler, IngestOutcomeHandler)
        assert handlers.ingest_outcome_handler.database is mock_db
    
    def test_creates_game_data_handler(self):
        """Creates GameDataHandler instance."""
        mock_db = MagicMock()
        handlers = Handlers(mock_db)
        
        assert isinstance(handlers.game_data_handler, GameDataHandler)
        # GameDataHandler stores database as _database (private)
        assert handlers.game_data_handler._database is mock_db
    
    def test_creates_all_score_handlers(self):
        """Creates all score handlers."""
        mock_db = MagicMock()
        handlers = Handlers(mock_db)
        
        assert handlers.odds_score_handler is not None
        assert handlers.outcome_score_handler is not None
        assert handlers.main_score_handler is not None
    
    def test_creates_all_core_handlers(self):
        """Creates all core chain handlers."""
        mock_db = MagicMock()
        handlers = Handlers(mock_db)
        
        assert handlers.set_weights_handler is not None
        assert handlers.miner_management_handler is not None
        assert handlers.sync_metagraph_handler is not None
    
    def test_all_handlers_share_same_database(self):
        """All handlers share the same database instance."""
        mock_db = MagicMock()
        handlers = Handlers(mock_db)
        
        assert handlers.ingest_odds_handler.database is mock_db
        assert handlers.ingest_outcome_handler.database is mock_db
        # GameDataHandler stores database as _database (private)
        assert handlers.game_data_handler._database is mock_db

