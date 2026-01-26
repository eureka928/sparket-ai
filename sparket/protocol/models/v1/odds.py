"""API v1 odds request/response models.

Client-submitted odds map to `miner_submission` rows via
`sparket.protocol.mapping.v1.map_submit_odds_to_miner_submission_rows`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import APIVersion, MarketKind, PriceSide, ResponseMeta


class OutcomePrice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    side: PriceSide
    odds_eu: float = Field(..., gt=1.0, le=1000.0, description="Decimal odds in (1, 1000]")
    imp_prob: float = Field(..., gt=0.0, lt=1.0, description="Raw implied prob before normalization")
    imp_prob_norm: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MarketSubmission(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_id: int
    kind: MarketKind
    priced_at: Optional[datetime] = None
    prices: List[OutcomePrice]

    @model_validator(mode="after")
    def _validate_prices_for_kind(self) -> "MarketSubmission":
        sides = {p.side for p in self.prices}
        if self.kind == "moneyline":
            # Must include both home and away; draw optional
            if not ({"home", "away"} <= {s.value if hasattr(s, 'value') else s for s in sides}):
                raise ValueError("moneyline prices must include both home and away sides")
        if self.kind == "total":
            if not ({"over", "under"} <= {s.value if hasattr(s, 'value') else s for s in sides}):
                raise ValueError("total prices must include both over and under")
        if self.kind == "spread":
            if not ({"home", "away"} <= {s.value if hasattr(s, 'value') else s for s in sides}):
                raise ValueError("spread prices must include both home and away")
        return self


class SubmitOddsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: APIVersion = Field(default=APIVersion.V1)
    miner_id: int
    miner_hotkey: str
    submissions: List[MarketSubmission]

    @model_validator(mode="after")
    def _validate_prob_ranges(self) -> "SubmitOddsRequest":
        for sub in self.submissions:
            for p in sub.prices:
                if p.imp_prob <= 0.0 or p.imp_prob >= 1.0:
                    raise ValueError("imp_prob must be in (0,1)")
                if p.imp_prob_norm is not None and not (0.0 <= p.imp_prob_norm <= 1.0):
                    raise ValueError("imp_prob_norm must be in [0,1]")
        return self


class SubmitOddsResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_id: int
    accepted: bool
    deduped: bool
    message: Optional[str] = None


class SubmitOddsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meta: ResponseMeta
    results: List[SubmitOddsResult]


__all__ = [
    "OutcomePrice",
    "MarketSubmission",
    "SubmitOddsRequest",
    "SubmitOddsResult",
    "SubmitOddsResponse",
]


