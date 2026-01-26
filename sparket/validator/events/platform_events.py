from __future__ import annotations

from typing import Any, Dict

from sparket.validator.events.event import Event
# Placeholder for sparket platform events. 

class PlatformEvent(Event):
    def __init__(self, *, kind: str, payload: Dict[str, Any]):
        super().__init__(event_id=None, event_type=f"platform.{kind}", event_data=payload)
