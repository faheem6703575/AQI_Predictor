"""Configuration loader.

Reads ``config.yaml`` and exposes a ``Config`` facade. Supports both
the legacy single-city schema (``city:``) and the new multi-city schema
(``cities:``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)
    _active_city: Optional[str] = None

    # ---- cities ----------------------------------------------------------
    @property
    def cities(self) -> List[Dict[str, Any]]:
        if "cities" in self.raw:
            return list(self.raw["cities"])
        if "city" in self.raw:
            return [self.raw["city"]]
        raise KeyError("config.yaml must define either 'city' or 'cities'")

    @property
    def default_city_name(self) -> str:
        if self.raw.get("default_city"):
            return self.raw["default_city"]
        return self.cities[0]["name"]

    def use_city(self, name: str) -> "Config":
        """Return the same Config but with a specific active city."""
        names = [c["name"] for c in self.cities]
        if name not in names:
            raise KeyError(f"Unknown city '{name}'. Known: {names}")
        clone = Config(raw=self.raw, _active_city=name)
        return clone

    @property
    def city(self) -> Dict[str, Any]:
        """Currently active city (defaults to ``default_city``)."""
        target = self._active_city or self.default_city_name
        for c in self.cities:
            if c["name"] == target:
                return c
        raise KeyError(target)

    # ---- other sections --------------------------------------------------
    @property
    def api(self) -> Dict[str, Any]:
        return self.raw["api"]

    @property
    def variables(self) -> Dict[str, List[str]]:
        return self.raw["variables"]

    @property
    def horizons(self) -> List[int]:
        return list(self.raw["forecast"]["horizons_hours"])

    @property
    def quantiles(self) -> List[float]:
        return list(self.raw["forecast"].get("quantiles", []))

    @property
    def target(self) -> str:
        return self.raw["forecast"]["target_variable"]

    @property
    def storage(self) -> Dict[str, str]:
        return self.raw["storage"]

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw["training"]

    @property
    def tuning(self) -> Dict[str, Any]:
        return self.raw["training"].get("tuning", {"enabled": False})

    @property
    def drift(self) -> Dict[str, Any]:
        return self.raw.get("drift", {"window_days": 14, "psi_alert_threshold": 0.25})

    @property
    def api_server(self) -> Dict[str, Any]:
        return self.raw.get("api_server", {"host": "0.0.0.0", "port": 8000})

    @property
    def alerts(self) -> Dict[str, Any]:
        return self.raw["alerts"]

    # ---- path helpers ----------------------------------------------------
    def path(self, key: str) -> Path:
        p = Path(self.storage[key])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(raw=data)
