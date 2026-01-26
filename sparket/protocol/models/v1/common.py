"""API v1 common models and error envelope.

These models are public contract shapes. They intentionally import shared
domain enums for consistency across API and provider integrations.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class APIVersion(str, Enum):
    V1 = "v1"


from sparket.shared.enums import MarketKind, PriceSide


class APIError(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str = Field(..., description="Stable machine-readable error code")
    message: str = Field(..., description="Human-friendly error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional structured error details"
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: APIVersion = Field(default=APIVersion.V1)
    error: APIError


class ResponseMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: APIVersion = Field(default=APIVersion.V1)
    request_id: Optional[str] = Field(
        default=None, description="Idempotency/request identifier for tracing"
    )


__all__ = [
    "APIVersion",
    "MarketKind",
    "PriceSide",
    "APIError",
    "ErrorResponse",
    "ResponseMeta",
]


