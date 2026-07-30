"""Inference: 72-hour AQI forecast with prediction intervals.

For each city we:
  1. Load the latest *point* model (best RMSE winner) and the latest
     *quantile* model (LightGBM quantile regression) from the registry.
  2. Grab the most recent feature-complete row from the feature store.
  3. Produce the horizon predictions + a smooth hourly curve + P10/P90
     bands if the quantile model is available.
  4. Append the forecast to ``data/forecast_history/`` for later drift
     evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config, load_config
from .feature_store import FeatureStore
from .forecast_history import append_forecast
from .model_registry import ModelRegistry
from .utils import aqi_category, get_logger

_LOG = get_logger("predict")


@dataclass
class Forecast:
    issued_at: pd.Timestamp
    city: str
    horizons: List[int]
    horizon_predictions: pd.DataFrame
    hourly_curve: pd.DataFrame
    model_name: str
    model_version: str
    quantiles: Dict[float, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issued_at": self.issued_at.isoformat(),
            "city": self.city,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "horizons": self.horizons,
            "horizon_predictions": self.horizon_predictions.to_dict(orient="records"),
            "hourly_curve": self.hourly_curve.to_dict(orient="records"),
            "quantiles": {str(q): arr.tolist() for q, arr in self.quantiles.items()},
        }


def _latest_feature_row(cfg: Config) -> pd.Series:
    fs = FeatureStore(cfg.path("feature_store_dir"))
    df = fs.group("engineered_features").read()
    if df.empty:
        raise RuntimeError("Feature store empty - run the feature pipeline first")
    if "city" in df.columns:
        df = df[df["city"] == cfg.city["name"]]
    df = df.sort_values("timestamp")
    target_cols = [c for c in df.columns if c.startswith("target_h")]
    feature_df = df.drop(columns=target_cols, errors="ignore")
    numeric = feature_df.select_dtypes(include="number")
    full_rows = numeric.dropna()
    if full_rows.empty:
        raise RuntimeError("No feature-complete rows available for inference")
    idx = full_rows.index[-1]
    return feature_df.loc[idx]


def predict_forecast(
    cfg: Optional[Config] = None,
    *,
    city: Optional[str] = None,
    log_history: bool = True,
) -> Forecast:
    cfg = cfg or load_config()
    if city:
        cfg = cfg.use_city(city)

    registry = ModelRegistry(cfg.path("model_registry_dir"))
    bundle = registry.load_latest(city=cfg.city["name"]) or registry.load_latest()
    if bundle is None:
        raise RuntimeError("No model in registry. Run the training pipeline first.")

    model = bundle["model"]
    meta = bundle["metadata"]
    features = meta["features"]
    horizons = meta["horizons"]
    target = meta["target"]

    latest_row = _latest_feature_row(cfg)
    missing = [f for f in features if f not in latest_row.index]
    if missing:
        raise RuntimeError(f"Feature drift - missing: {missing[:5]}")

    X = pd.DataFrame([latest_row[features]])
    X.attrs["target_col"] = target
    y_pred = model.predict(X)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    aqi_values = y_pred[0].tolist()

    # Quantile predictions (if the quantile model is registered)
    quantiles_arr: Dict[float, np.ndarray] = {}
    quantile_bundle = registry.load_quantile(city=cfg.city["name"])
    if quantile_bundle is not None:
        q_model = quantile_bundle["model"]
        try:
            q_preds = q_model.predict_quantiles(X)  # {q: (1, n_h)}
            quantiles_arr = {q: q_preds[q][0] for q in q_preds}
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Quantile inference failed: %s", exc)

    issued_at = pd.to_datetime(latest_row["timestamp"], utc=True)
    horizon_rows = []
    for i, (h, aqi) in enumerate(zip(horizons, aqi_values)):
        cat = aqi_category(aqi)
        row = {
            "timestamp": issued_at + pd.Timedelta(hours=int(h)),
            "horizon_h": int(h),
            "aqi": float(aqi),
            "category": cat["category"],
            "color": cat["color"],
        }
        for q, arr in quantiles_arr.items():
            row[f"q{int(q*100):02d}"] = float(arr[i])
        horizon_rows.append(row)
    horizon_df = pd.DataFrame(horizon_rows)

    # Smooth hourly curve
    max_h = max(horizons)
    future_index = pd.date_range(
        start=issued_at + pd.Timedelta(hours=1),
        end=issued_at + pd.Timedelta(hours=max_h),
        freq="h", tz="UTC",
    )
    anchor_hours = np.array(horizons, dtype=float)
    anchor_vals = np.array(aqi_values, dtype=float)
    hours_ahead = np.arange(1, max_h + 1, dtype=float)
    hourly_curve = pd.DataFrame({"timestamp": future_index})
    hourly_curve["aqi"] = np.interp(hours_ahead, anchor_hours, anchor_vals)
    if quantiles_arr:
        for q, arr in quantiles_arr.items():
            hourly_curve[f"q{int(q*100):02d}"] = np.interp(hours_ahead, anchor_hours, arr)
    cats = hourly_curve["aqi"].apply(aqi_category)
    hourly_curve["category"] = cats.apply(lambda d: d["category"])
    hourly_curve["color"] = cats.apply(lambda d: d["color"])

    fc = Forecast(
        issued_at=issued_at,
        city=cfg.city["name"],
        horizons=horizons,
        horizon_predictions=horizon_df,
        hourly_curve=hourly_curve,
        model_name=meta["name"],
        model_version=meta["version"],
        quantiles=quantiles_arr,
    )

    if log_history:
        try:
            append_forecast(
                cfg.path("forecast_history_dir"),
                city=cfg.city["name"],
                issued_at=issued_at,
                horizon_predictions=horizon_df,
                model_name=meta["name"],
                model_version=meta["version"],
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Forecast history logging failed: %s", exc)

    _LOG.info(
        "Forecast %s @ %s using %s@%s: max(AQI)=%.1f%s",
        cfg.city["name"], issued_at, meta["name"], meta["version"],
        horizon_df["aqi"].max(),
        " (with P10/P90)" if quantiles_arr else "",
    )
    return fc


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--city", type=str, default=None)
    args = p.parse_args()
    fc = predict_forecast(city=args.city)
    print(fc.horizon_predictions.to_string(index=False))
