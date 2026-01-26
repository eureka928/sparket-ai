"""API v1 outcome submission models.

Outcome envelopes are written to the `inbox` with a dedupe key via
`sparket.protocol.mapping.v1.map_submit_outcome_to_inbox_row`.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import APIVersion, ResponseMeta


class OutcomeEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str
    source_type: str = Field(description="league_api | official_site | aggregator")
    captured_at: datetime
    hash: Optional[str] = None


class SubmitOutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: APIVersion = Field(default=APIVersion.V1)
    event_id: str
    miner_hotkey: str
    winner_label: str = Field(description="home | away | draw")
    final_score: str = Field(description="e.g., 102-98")
    ts_submit: datetime
    sources: List[OutcomeEvidence]


class SubmitOutcomeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meta: ResponseMeta
    accepted: bool
    deduped: bool
    message: Optional[str] = None


__all__ = [
    "OutcomeEvidence",
    "SubmitOutcomeRequest",
    "SubmitOutcomeResponse",
]


