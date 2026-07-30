"""Hazardous-AQI alerts.

This module checks a forecast against the configured thresholds and emits
alerts via:
  * console / logging   (always)
  * a JSON file ``reports/alerts.json``
  * a webhook URL set in ``ALERT_WEBHOOK_URL``  (optional, e.g. Slack/Discord)

We intentionally keep this dependency-free so it works inside GitHub Actions
without secrets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .config import Config, load_config
from .utils import aqi_category, get_logger

_LOG = get_logger("alerts")


LEVEL_ORDER = [
    "good", "moderate", "unhealthy_sensitive",
    "unhealthy", "very_unhealthy", "hazardous",
]


def _level_from_aqi(aqi: float, thresholds: Dict[str, float]) -> str:
    if aqi >= thresholds.get("hazardous", 301):
        return "hazardous"
    if aqi >= thresholds.get("very_unhealthy", 201):
        return "very_unhealthy"
    if aqi >= thresholds.get("unhealthy", 151):
        return "unhealthy"
    if aqi >= thresholds.get("unhealthy_sensitive", 101):
        return "unhealthy_sensitive"
    if aqi >= thresholds.get("moderate", 51):
        return "moderate"
    return "good"


@dataclass
class Alert:
    issued_at: str
    city: str
    level: str
    aqi: float
    timestamp: str
    category: str
    color: str
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)


def evaluate_forecast_for_alerts(
    horizon_predictions: pd.DataFrame,
    cfg: Optional[Config] = None,
) -> List[Alert]:
    cfg = cfg or load_config()
    cfg_alerts = cfg.alerts
    thresholds = cfg_alerts.get("thresholds", {})
    min_level = cfg_alerts.get("notify_on_level", "unhealthy")
    min_idx = LEVEL_ORDER.index(min_level)

    alerts: List[Alert] = []
    issued = datetime.now(timezone.utc).isoformat()
    for _, row in horizon_predictions.iterrows():
        level = _level_from_aqi(row["aqi"], thresholds)
        if LEVEL_ORDER.index(level) < min_idx:
            continue
        cat = aqi_category(row["aqi"])
        alerts.append(
            Alert(
                issued_at=issued,
                city=cfg.city["name"],
                level=level,
                aqi=float(row["aqi"]),
                timestamp=pd.Timestamp(row["timestamp"]).isoformat(),
                category=cat["category"],
                color=cat["color"],
                message=(
                    f"AQI in {cfg.city['name']} predicted to reach {row['aqi']:.0f} "
                    f"({cat['category']}) at {pd.Timestamp(row['timestamp']).strftime('%Y-%m-%d %H:%MZ')}."
                ),
                extra={"horizon_h": int(row.get("horizon_h", -1))},
            )
        )
    return alerts


def _post_webhook(alerts: List[Alert], url: str) -> None:
    payload = {
        "content": "**AQI Alert**\n" + "\n".join(
            f"- [{a.level.upper()}] {a.message}" for a in alerts
        )
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        _LOG.info("Posted %d alerts to webhook", len(alerts))
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("Webhook delivery failed: %s", exc)


def dispatch(alerts: List[Alert], cfg: Optional[Config] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    out_dir = cfg.path("reports_dir")
    out_path = out_dir / "alerts.json"

    payload = {
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "count": len(alerts),
        "alerts": [a.__dict__ for a in alerts],
    }
    out_path.write_text(json.dumps(payload, indent=2))

    if alerts:
        _LOG.warning("=== %d hazardous AQI alerts ===", len(alerts))
        for a in alerts:
            _LOG.warning("[%s] %s", a.level.upper(), a.message)

    url = os.environ.get("ALERT_WEBHOOK_URL")
    if url and alerts:
        _post_webhook(alerts, url)
    return payload
