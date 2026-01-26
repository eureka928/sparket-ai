from __future__ import annotations

import os
from typing import Any, Dict

import yaml
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from sparket.config import Settings, load_settings
from sparket.config.core import last_yaml_path


class CadenceSettings(BaseModel):
    odds_seconds: int = 60
    outcomes_seconds: int = 120


class RateSettings(BaseModel):
    global_per_minute: int = 60
    per_market_per_minute: int = 6


class RetrySettings(BaseModel):
    max_attempts: int = 3
    initial_backoff_ms: int = 250
    max_backoff_ms: int = 2000


class IdempotencySettings(BaseModel):
    bucket_seconds: int = 60


class EndpointOverride(BaseModel):
    url: str | None = None
    host: str | None = None
    port: int | None = None


class LoggingSettings(BaseModel):
    json_logs: bool = True
    level: str = "INFO"

    @model_validator(mode="before")
    @classmethod
    def _alias_json(cls, data: dict[str, object]) -> dict[str, object]:
        if isinstance(data, dict) and "json" in data and "json_logs" not in data:
            data = dict(data)
            data["json_logs"] = data.pop("json")
        return data


class MinerSettings(BaseModel):
    database_filename: str = "miner.db"
    id: int = 0
    markets: list[int] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    cadence: CadenceSettings = Field(default_factory=CadenceSettings)
    rate: RateSettings = Field(default_factory=RateSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    idempotency: IdempotencySettings = Field(default_factory=IdempotencySettings)
    allow_connection_info_from_unpermitted_validators: bool = False
    endpoint_override: EndpointOverride = Field(default_factory=EndpointOverride)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def database_path(self, test_mode: bool = False) -> str:
        """Return the full path to the miner database file."""
        from sparket.config import _data_dir
        import os
        data_dir = _data_dir(test_mode)
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, self.database_filename)

    @model_validator(mode="before")
    @classmethod
    def _apply_yaml_overrides(cls, data: Any) -> Any:
        overrides = _load_miner_yaml_overrides()
        if not overrides:
            return data
        merged: Dict[str, Any] = dict(overrides)
        if isinstance(data, dict):
            merged.update(data)
        return merged


class Config(BaseSettings):
    core: Settings = Field(default_factory=lambda: load_settings(role="miner"))
    miner: MinerSettings = Field(default_factory=MinerSettings)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_miner_yaml_overrides() -> Dict[str, Any]:
    candidates: list[Path] = []
    yaml_path = last_yaml_path()
    if yaml_path:
        candidates.append(Path(yaml_path).resolve())
    package_default = _repo_root() / "config" / "miner.yaml"
    candidates.append(package_default)

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            continue
        miner_section = data.get("miner")
        if isinstance(miner_section, dict):
            return miner_section
    return {}


