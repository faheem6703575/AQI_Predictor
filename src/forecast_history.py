"""Forecast history log.

Every time we issue a forecast, we append a row to
``data/forecast_history/<city>.parquet`` with the issue-time, target
timestamp, predicted AQI, and the model+version used. Once the true
observation catches up (`hours_ahead` hours later), the drift monitor
joins those rows against the actuals and computes error over time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .utils import get_logger

_LOG = get_logger("forecast_history")


def _table_path(root: Path, city: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{city.replace(' ', '_')}.parquet"


def append_forecast(
    root: Path,
    *,
    city: str,
    issued_at: pd.Timestamp,
    horizon_predictions: pd.DataFrame,
    model_name: str,
    model_version: str,
    quantiles: Optional[dict] = None,
) -> Path:
    """Append one row per horizon to the city's forecast history."""
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for _, r in horizon_predictions.iterrows():
        row = {
            "logged_at": now,
            "issued_at": pd.Timestamp(issued_at).isoformat(),
            "target_timestamp": pd.Timestamp(r["timestamp"]).isoformat(),
            "horizon_h": int(r["horizon_h"]),
            "predicted_aqi": float(r["aqi"]),
            "model_name": model_name,
            "model_version": model_version,
            "city": city,
        }
        if quantiles:
            for q, arr in quantiles.items():
                row[f"predicted_aqi_q{int(q*100):02d}"] = float(arr[int(r["horizon_h"])])
        rows.append(row)
    new_df = pd.DataFrame(rows)

    path = _table_path(root, city)
    if path.exists():
        existing = pd.read_parquet(path)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_parquet(path, index=False)
    _LOG.info("Appended %d forecast rows for %s", len(rows), city)
    return path


def read_history(root: Path, city: str) -> pd.DataFrame:
    path = _table_path(root, city)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["issued_at"] = pd.to_datetime(df["issued_at"], utc=True)
    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"], utc=True)
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
    return df


def list_cities(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted({p.stem.replace("_", " ") for p in root.glob("*.parquet")})
