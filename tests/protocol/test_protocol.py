"""Tests for protocol/protocol.py - Synapse types and protocol."""

from __future__ import annotations

import pytest

from sparket.protocol.protocol import (
    SparketSynapseType,
    SparketSynapse,
)


class TestSparketSynapseType:
    """Tests for SparketSynapseType enum."""
    
    def test_odds_push_value(self):
        """ODDS_PUSH has correct value."""
        assert SparketSynapseType.ODDS_PUSH.value == "odds_push"
    
    def test_outcome_push_value(self):
        """OUTCOME_PUSH has correct value."""
        assert SparketSynapseType.OUTCOME_PUSH.value == "outcome_push"
    
    def test_game_data_request_value(self):
        """GAME_DATA_REQUEST has correct value."""
        assert SparketSynapseType.GAME_DATA_REQUEST.value == "game_data_request"
    
    def test_connection_info_push_value(self):
        """CONNECTION_INFO_PUSH has correct value."""
        assert SparketSynapseType.CONNECTION_INFO_PUSH.value == "connection_info_push"
    
    def test_enum_is_string_enum(self):
        """Enum values are strings."""
        for stype in SparketSynapseType:
            assert isinstance(stype.value, str)


class TestSparketSynapse:
    """Tests for SparketSynapse class."""
    
    def test_default_type(self):
        """Default type is CONNECTION_INFO_PUSH."""
        synapse = SparketSynapse()
        assert synapse.type == SparketSynapseType.CONNECTION_INFO_PUSH
    
    def test_default_payload_is_dict(self):
        """Default payload is empty dict."""
        synapse = SparketSynapse()
        assert synapse.payload == {}
    
    def test_creates_with_type_and_payload(self):
        """Creates synapse with specified type and payload."""
        synapse = SparketSynapse(
            type=SparketSynapseType.ODDS_PUSH,
            payload={"submissions": []}
        )
        
        assert synapse.type == SparketSynapseType.ODDS_PUSH
        assert synapse.payload == {"submissions": []}
    
    def test_accepts_string_type(self):
        """Accepts string as type."""
        synapse = SparketSynapse(type="odds_push", payload={})
        assert synapse.type == "odds_push"
    
    def test_coerce_json_converts_enum(self):
        """_coerce_json converts enum to string value."""
        result = SparketSynapse._coerce_json(SparketSynapseType.ODDS_PUSH)
        assert result == "odds_push"
    
    def test_coerce_json_handles_dict(self):
        """_coerce_json recursively handles dicts."""
        result = SparketSynapse._coerce_json({
            "type": SparketSynapseType.OUTCOME_PUSH,
            "data": {"nested": SparketSynapseType.GAME_DATA_REQUEST}
        })
        
        assert result["type"] == "outcome_push"
        assert result["data"]["nested"] == "game_data_request"
    
    def test_coerce_json_handles_list(self):
        """_coerce_json recursively handles lists."""
        result = SparketSynapse._coerce_json([
            SparketSynapseType.ODDS_PUSH,
            SparketSynapseType.OUTCOME_PUSH,
        ])
        
        assert result == ["odds_push", "outcome_push"]
    
    def test_coerce_json_preserves_primitives(self):
        """_coerce_json preserves primitive values."""
        assert SparketSynapse._coerce_json("string") == "string"
        assert SparketSynapse._coerce_json(123) == 123
        assert SparketSynapse._coerce_json(True) is True
        assert SparketSynapse._coerce_json(None) is None
    
    def test_model_dump_converts_type(self):
        """model_dump converts enum type to string."""
        synapse = SparketSynapse(type=SparketSynapseType.ODDS_PUSH, payload={})
        data = synapse.model_dump()
        
        assert data["type"] == "odds_push"
    
    def test_model_dump_converts_payload_enums(self):
        """model_dump converts enums in payload."""
        synapse = SparketSynapse(
            type=SparketSynapseType.CONNECTION_INFO_PUSH,
            payload={"nested_type": SparketSynapseType.ODDS_PUSH}
        )
        data = synapse.model_dump()
        
        assert data["payload"]["nested_type"] == "odds_push"
    
    def test_dict_is_alias_for_model_dump(self):
        """dict() returns same as model_dump()."""
        synapse = SparketSynapse(type=SparketSynapseType.ODDS_PUSH, payload={"key": "value"})
        
        assert synapse.dict() == synapse.model_dump()
    
    def test_serialize_returns_model_dump(self):
        """serialize() returns model_dump()."""
        synapse = SparketSynapse(type=SparketSynapseType.ODDS_PUSH, payload={"key": "value"})
        
        assert synapse.serialize() == synapse.model_dump()
    
    def test_deserialize_returns_payload(self):
        """deserialize() returns payload."""
        synapse = SparketSynapse(
            type=SparketSynapseType.ODDS_PUSH,
            payload={"submissions": [{"market_id": 100}]}
        )
        
        result = synapse.deserialize()
        
        assert result == {"submissions": [{"market_id": 100}]}
    
    def test_payload_can_be_complex(self):
        """Payload can contain complex nested structures."""
        complex_payload = {
            "submissions": [
                {
                    "market_id": 100,
                    "prices": [
                        {"side": "home", "odds_eu": 1.91},
                        {"side": "away", "odds_eu": 2.05},
                    ]
                }
            ],
            "metadata": {
                "timestamp": "2025-12-08T14:30:00Z",
                "version": 1
            }
        }
        
        synapse = SparketSynapse(
            type=SparketSynapseType.ODDS_PUSH,
            payload=complex_payload
        )
        
        assert synapse.payload == complex_payload
        assert synapse.deserialize() == complex_payload

