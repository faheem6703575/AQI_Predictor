"""Open-Meteo data fetcher.

Wraps three free Open-Meteo endpoints (no API key required):
  * Air Quality        - hourly PM/NO2/SO2/O3/CO/AQI (past + short forecast)
  * Historical Weather - hourly ERA5 archive back many years
  * Weather Forecast   - hourly weather forecast for the next few days

All returned frames have a UTC ``timestamp`` column and a ``city`` column so
they can be concatenated across cities and de-duplicated in the store.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import Config
from .utils import get_logger, http_get_json

_LOG = get_logger("data_fetcher")


def _utcify(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.rename(columns={time_col: "timestamp"})
    return df


def _tag_city(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["city"] = city_name
    return df


# ---------------------------------------------------------------------------
# Air quality (historical + rolling)
# ---------------------------------------------------------------------------
def fetch_air_quality(
    cfg: Config,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    past_days: Optional[int] = None,
    forecast_days: int = 0,
) -> pd.DataFrame:
    params: Dict[str, Any] = {
        "latitude": cfg.city["latitude"],
        "longitude": cfg.city["longitude"],
        "hourly": ",".join(cfg.variables["air_quality"]),
        "timezone": "UTC",
    }
    if start_date and end_date:
        params["start_date"] = start_date.isoformat()
        params["end_date"] = end_date.isoformat()
    else:
        if past_days is not None:
            params["past_days"] = past_days
        params["forecast_days"] = forecast_days

    data = http_get_json(
        cfg.api["air_quality_url"],
        params,
        timeout=cfg.api["timeout_seconds"],
        retries=cfg.api["retries"],
        backoff=cfg.api["retry_backoff_seconds"],
    )
    hourly = data.get("hourly") or {}
    if not hourly:
        _LOG.warning("Empty air-quality response for %s", cfg.city["name"])
        return pd.DataFrame()
    return _tag_city(_utcify(pd.DataFrame(hourly)), cfg.city["name"])


# ---------------------------------------------------------------------------
# Weather (archive + forecast)
# ---------------------------------------------------------------------------
def fetch_weather_history(
    cfg: Config,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    params = {
        "latitude": cfg.city["latitude"],
        "longitude": cfg.city["longitude"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(cfg.variables["weather"]),
        "timezone": "UTC",
    }
    data = http_get_json(
        cfg.api["archive_url"],
        params,
        timeout=cfg.api["timeout_seconds"],
        retries=cfg.api["retries"],
        backoff=cfg.api["retry_backoff_seconds"],
    )
    hourly = data.get("hourly") or {}
    if not hourly:
        _LOG.warning("Empty weather-archive response for %s", cfg.city["name"])
        return pd.DataFrame()
    return _tag_city(_utcify(pd.DataFrame(hourly)), cfg.city["name"])


def fetch_weather_forecast(cfg: Config, *, past_days: int = 2, forecast_days: int = 3) -> pd.DataFrame:
    params = {
        "latitude": cfg.city["latitude"],
        "longitude": cfg.city["longitude"],
        "hourly": ",".join(cfg.variables["weather"]),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    data = http_get_json(
        cfg.api["forecast_url"],
        params,
        timeout=cfg.api["timeout_seconds"],
        retries=cfg.api["retries"],
        backoff=cfg.api["retry_backoff_seconds"],
    )
    hourly = data.get("hourly") or {}
    if not hourly:
        _LOG.warning("Empty weather-forecast response for %s", cfg.city["name"])
        return pd.DataFrame()
    return _tag_city(_utcify(pd.DataFrame(hourly)), cfg.city["name"])


# ---------------------------------------------------------------------------
# Combined fetchers
# ---------------------------------------------------------------------------
def _merge_on_time(*frames: pd.DataFrame) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        common = [c for c in ("timestamp", "city") if c in out.columns and c in f.columns]
        out = out.merge(f, on=common, how="outer")
    return out.sort_values("timestamp").reset_index(drop=True)


def fetch_combined_history(
    cfg: Config,
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    aq = fetch_air_quality(cfg, start_date=start_date, end_date=end_date)
    wx = fetch_weather_history(cfg, start_date=start_date, end_date=end_date)
    return _merge_on_time(aq, wx)


def fetch_combined_recent(
    cfg: Config,
    *,
    past_days: int = 7,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """Air-quality + weather (past ``past_days`` + upcoming ``forecast_days``).

    The forecast portion is what gives us the "future weather" columns used
    as exogenous features when predicting AQI multiple hours ahead.
    """
    aq = fetch_air_quality(cfg, past_days=past_days, forecast_days=forecast_days)
    wx = fetch_weather_forecast(cfg, past_days=past_days, forecast_days=forecast_days)
    return _merge_on_time(aq, wx)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def days_ago(n: int) -> date:
    return (utc_now() - timedelta(days=n)).date()
