"""Feature pipeline (advanced).

Produces model-ready features from raw weather + air-quality observations
and stores them per-city in the local feature store.

Key advances over v1:
  * **Multi-city** - the pipeline loops over every city in the config, and
    the feature store partitions rows by ``city``.
  * **Future-weather features** - for each forecast horizon ``h`` we add
    ``<weather_var>_fh<h>``, i.e. the weather forecast at t+h known at
    prediction time. These "future exogenous" features materially improve
    24/48/72 h AQI forecasts because tomorrow's wind + rain change the AQI
    trajectory.
  * **Missingness flags** - for every raw variable we add a ``<var>_isna``
    indicator (0/1) so the model can learn from imputation patterns.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data_fetcher import (
    days_ago,
    fetch_combined_history,
    fetch_combined_recent,
    fetch_weather_forecast,
)
from .feature_store import FeatureStore
from .utils import get_logger

_LOG = get_logger("feature_pipeline")

LAGS = [1, 3, 6, 12, 24, 48, 72]
ROLLINGS = [6, 24]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def _clean(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=[c for c in ["timestamp", "city"] if c in df.columns]
    ).reset_index(drop=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df = df.dropna(how="all", subset=numeric_cols)

    # Missingness flags before we impute
    for col in numeric_cols:
        df[f"{col}_isna"] = df[col].isna().astype(int)

    df[numeric_cols] = (
        df[numeric_cols]
        .interpolate(method="linear", limit_direction="both", limit=6)
        .bfill()
        .ffill()
    )
    if target in df.columns:
        df = df.dropna(subset=[target]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"].dt
    df["hour"] = ts.hour.astype(int)
    df["dayofweek"] = ts.dayofweek.astype(int)
    df["day"] = ts.day.astype(int)
    df["month"] = ts.month.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_lags(df: pd.DataFrame, target: str) -> pd.DataFrame:
    for lag in LAGS:
        df[f"{target}_lag_{lag}h"] = df[target].shift(lag)
    return df


def _add_rolling(df: pd.DataFrame, target: str) -> pd.DataFrame:
    for w in ROLLINGS:
        df[f"{target}_rmean_{w}h"] = df[target].shift(1).rolling(window=w, min_periods=2).mean()
        df[f"{target}_rstd_{w}h"] = df[target].shift(1).rolling(window=w, min_periods=2).std()
    return df


def _add_change_rate(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df[f"{target}_diff_1h"] = df[target].diff(1)
    df[f"{target}_pct_change_1h"] = (
        df[target].pct_change(1).replace([np.inf, -np.inf], 0).fillna(0)
    )
    df[f"{target}_diff_24h"] = df[target].diff(24)
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if {"temperature_2m", "relative_humidity_2m"}.issubset(df.columns):
        df["temp_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100.0
    if {"wind_speed_10m", "wind_direction_10m"}.issubset(df.columns):
        rad = np.deg2rad(df["wind_direction_10m"].fillna(0))
        df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
        df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)
    return df


def _add_future_weather(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """For each horizon h in cfg.horizons and each variable in
    cfg.variables['future_weather'], add ``<var>_fh<h>`` = value h hours
    ahead of the current row. This lets the model exploit the fact that
    at inference we already know Open-Meteo's weather forecast at t+h.
    """
    future_vars = cfg.variables.get("future_weather", []) or []
    horizons = cfg.horizons
    for h in horizons:
        for var in future_vars:
            if var in df.columns:
                df[f"{var}_fh{h}"] = df[var].shift(-h)
    return df


def engineer(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if df.empty:
        return df
    target = cfg.target
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df = _add_calendar(df)
    df = _add_lags(df, target)
    df = _add_rolling(df, target)
    df = _add_change_rate(df, target)
    df = _add_interactions(df)
    df = _add_future_weather(df, cfg)
    return df


def add_targets(df: pd.DataFrame, target: str, horizons: List[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"target_h{h}"] = out[target].shift(-h)
    return out


# ---------------------------------------------------------------------------
# Entry points (per city + multi-city)
# ---------------------------------------------------------------------------
def _process_city(cfg_city: Config, past_days: int) -> pd.DataFrame:
    """Hourly refresh for a single city."""
    target = cfg_city.target
    fs = FeatureStore(cfg_city.path("feature_store_dir"))
    raw_group = fs.group("raw_observations")
    feat_group = fs.group("engineered_features")

    _LOG.info("[%s] fetching recent (past_days=%d)", cfg_city.city["name"], past_days)
    recent = fetch_combined_recent(cfg_city, past_days=past_days, forecast_days=3)
    recent = _clean(recent, target=target)
    if recent.empty:
        _LOG.warning("[%s] no recent data - skipping", cfg_city.city["name"])
        return recent
    raw_group.upsert(recent)

    full_raw = raw_group.read()
    full_raw = full_raw[full_raw.get("city", cfg_city.city["name"]) == cfg_city.city["name"]]
    full_raw = _clean(full_raw, target=target)
    engineered = engineer(full_raw, cfg_city)
    engineered = add_targets(engineered, target=target, horizons=cfg_city.horizons)
    engineered["city"] = cfg_city.city["name"]
    feat_group.upsert(engineered)
    return engineered


def run_feature_pipeline(
    cfg: Optional[Config] = None,
    *,
    past_days: int = 14,
    cities: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run the hourly refresh for every city in ``cities`` (or all of them)."""
    cfg = cfg or load_config()
    city_names = cities or [c["name"] for c in cfg.cities]
    all_frames: list[pd.DataFrame] = []
    for name in city_names:
        try:
            frame = _process_city(cfg.use_city(name), past_days=past_days)
            if not frame.empty:
                all_frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("[%s] hourly refresh failed: %s", name, exc)
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        _LOG.info(
            "Feature pipeline complete: %d rows across %d cities",
            len(combined), len(all_frames),
        )
        return combined
    return pd.DataFrame()


def _backfill_city(
    cfg_city: Config,
    *,
    backfill_days: int,
    chunk_days: int,
) -> pd.DataFrame:
    target = cfg_city.target
    fs = FeatureStore(cfg_city.path("feature_store_dir"))
    raw_group = fs.group("raw_observations")
    feat_group = fs.group("engineered_features")

    end = days_ago(1)
    start = end - timedelta(days=backfill_days)

    chunks: list[pd.DataFrame] = []
    cur_start = start
    while cur_start <= end:
        cur_end = min(cur_start + timedelta(days=chunk_days - 1), end)
        _LOG.info("[%s] backfill %s -> %s", cfg_city.city["name"], cur_start, cur_end)
        chunk = fetch_combined_history(cfg_city, start_date=cur_start, end_date=cur_end)
        if not chunk.empty:
            chunks.append(chunk)
        cur_start = cur_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    raw = pd.concat(chunks, ignore_index=True)
    raw = _clean(raw, target=target)
    raw_group.upsert(raw)

    full_raw = raw_group.read()
    full_raw = full_raw[full_raw.get("city", cfg_city.city["name"]) == cfg_city.city["name"]]
    full_raw = _clean(full_raw, target=target)
    engineered = engineer(full_raw, cfg_city)
    engineered = add_targets(engineered, target=target, horizons=cfg_city.horizons)
    engineered["city"] = cfg_city.city["name"]
    feat_group.upsert(engineered)
    return engineered


def run_backfill(
    cfg: Optional[Config] = None,
    *,
    backfill_days: Optional[int] = None,
    chunk_days: int = 90,
    cities: Optional[List[str]] = None,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    n_days = backfill_days or int(cfg.training.get("backfill_days", 365))
    city_names = cities or [c["name"] for c in cfg.cities]

    all_frames: list[pd.DataFrame] = []
    for name in city_names:
        try:
            frame = _backfill_city(
                cfg.use_city(name),
                backfill_days=n_days,
                chunk_days=chunk_days,
            )
            if not frame.empty:
                all_frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("[%s] backfill failed: %s", name, exc)
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        _LOG.info(
            "Backfill complete: %d engineered rows across %d cities",
            len(combined), len(all_frames),
        )
        return combined
    return pd.DataFrame()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Run the AQI feature pipeline")
    p.add_argument("--mode", choices=["hourly", "backfill"], default="hourly")
    p.add_argument("--past-days", type=int, default=14)
    p.add_argument("--backfill-days", type=int, default=None)
    p.add_argument("--city", type=str, default=None, help="Limit to a single city")
    args = p.parse_args()

    cities = [args.city] if args.city else None
    if args.mode == "hourly":
        run_feature_pipeline(past_days=args.past_days, cities=cities)
    else:
        run_backfill(backfill_days=args.backfill_days, cities=cities)
