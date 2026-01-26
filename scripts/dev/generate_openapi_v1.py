from __future__ import annotations

import json
from typing import Dict

from sparket.api.models.v1.common import ErrorResponse
from sparket.api.models.v1.odds import SubmitOddsRequest, SubmitOddsResponse
from sparket.api.models.v1.outcomes import SubmitOutcomeRequest, SubmitOutcomeResponse


def build_openapi() -> Dict:
    """Generate a minimal OpenAPI-like schema from Pydantic models.

    Note: For a full spec, wire FastAPI/Starlette routes. This generator
    captures component schemas and simple endpoints documentation.
    """
    components = {
        "schemas": {
            "SubmitOddsRequest": SubmitOddsRequest.model_json_schema(),
            "SubmitOddsResponse": SubmitOddsResponse.model_json_schema(),
            "SubmitOutcomeRequest": SubmitOutcomeRequest.model_json_schema(),
            "SubmitOutcomeResponse": SubmitOutcomeResponse.model_json_schema(),
            "ErrorResponse": ErrorResponse.model_json_schema(),
        }
    }
    paths = {
        "/api/v1/odds:submit": {
            "post": {
                "summary": "Submit miner odds",
                "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/SubmitOddsRequest"}}}},
                "responses": {
                    "200": {"description": "OK", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SubmitOddsResponse"}}}},
                    "400": {"description": "Bad Request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                },
            }
        },
        "/api/v1/outcomes:submit": {
            "post": {
                "summary": "Submit outcome evidence",
                "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/SubmitOutcomeRequest"}}}},
                "responses": {
                    "200": {"description": "OK", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SubmitOutcomeResponse"}}}},
                    "400": {"description": "Bad Request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}}},
                },
            }
        },
    }
    return {"openapi": "3.1.0", "info": {"title": "Sparket Subnet API", "version": "v1"}, "paths": paths, "components": components}


if __name__ == "__main__":
    spec = build_openapi()
    print(json.dumps(spec, indent=2))
