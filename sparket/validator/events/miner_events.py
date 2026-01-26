from __future__ import annotations

from typing import Any, Dict

from sparket.validator.events.event import Event
from sparket.protocol.protocol import SparketSynapseType


class MinerEvent(Event):
    """
    Base miner event with common fields (miner_hotkey, payload).
    Subclasses MUST provide concrete event_type and deterministic event_id.
    """

    def __init__(self, *, event_id: str | None, event_type: str, miner_hotkey: str, payload: Dict[str, Any]):
        super().__init__(
            event_id=event_id,
            event_type=event_type,
            event_data={"miner_hotkey": miner_hotkey, "payload": payload},
        )

    @staticmethod
    def ts_round(payload: Dict[str, Any]) -> str:
        val = payload.get("ts_round")
        return str(val) if val is not None else ""

    @staticmethod
    def canonical_payload(payload: Dict[str, Any]) -> str:
        return Event.canonical_json(payload)


class MinerOddsPushed(MinerEvent):
    def __init__(self, *, miner_hotkey: str, payload: Dict[str, Any]):
        # Prefer ts_round or canonical payload for idempotency
        ts_round = MinerEvent.ts_round(payload)
        ev_id = Event.make_id(
            SparketSynapseType.ODDS_PUSH.value,
            miner_hotkey,
            ts_round or MinerEvent.canonical_payload(payload),
        )
        super().__init__(
            event_id=ev_id,
            event_type=SparketSynapseType.ODDS_PUSH.value,
            miner_hotkey=miner_hotkey,
            payload=payload,
        )


class MinerOutcomePushed(MinerEvent):
    def __init__(self, *, miner_hotkey: str, payload: Dict[str, Any]):
        # Outcome submissions may not have ts_round; fall back to canonical payload
        ts_round = MinerEvent.ts_round(payload)
        ev_id = Event.make_id(
            SparketSynapseType.OUTCOME_PUSH.value,
            miner_hotkey,
            ts_round or MinerEvent.canonical_payload(payload),
        )
        super().__init__(
            event_id=ev_id,
            event_type=SparketSynapseType.OUTCOME_PUSH.value,
            miner_hotkey=miner_hotkey,
            payload=payload,
        )




