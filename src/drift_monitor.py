"""Model & data drift monitoring.

Two flavours of drift are checked:

* **Feature drift** - Population Stability Index (PSI) between a
  *reference* window (training tail) and a *recent* window (last N days
  in the feature store). A large PSI on a feature indicates its
  distribution has shifted since training - retraining may be required.

* **Prediction-error drift** - joins the ``forecast_history`` log with
  the ground-truth AQI from ``raw_observations`` and computes rolling
  MAE / RMSE per horizon. If recent error is significantly worse than
  the historical baseline, we emit a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config, load_config
from .feature_store import FeatureStore
from .forecast_history import read_history
from .model_registry import ModelRegistry
from .utils import get_logger

_LOG = get_logger("drift_monitor")


# ---------------------------------------------------------------------------
# PSI (Population Stability Index)
# ---------------------------------------------------------------------------
def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_pct = np.clip(e_hist / max(len(expected), 1), 1e-6, None)
    a_pct = np.clip(a_hist / max(len(actual), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


@dataclass
class DriftReport:
    city: str
    computed_at: str
    feature_psi: Dict[str, float]
    top_shifted_features: List[str]
    error_rolling: pd.DataFrame
    error_baseline: pd.DataFrame
    alert: bool
    alert_details: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "city": self.city,
            "computed_at": self.computed_at,
            "feature_psi": self.feature_psi,
            "top_shifted_features": self.top_shifted_features,
            "error_rolling": self.error_rolling.to_dict(orient="records"),
            "error_baseline": self.error_baseline.to_dict(orient="records"),
            "alert": self.alert,
            "alert_details": self.alert_details,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_drift(
    cfg: Optional[Config] = None,
    *,
    city: Optional[str] = None,
) -> DriftReport:
    cfg = cfg or load_config()
    if city:
        cfg = cfg.use_city(city)
    city_name = cfg.city["name"]

    # -- Feature drift --------------------------------------------------
    fs = FeatureStore(cfg.path("feature_store_dir"))
    feats = fs.group("engineered_features").read()
    if "city" in feats.columns:
        feats = feats[feats["city"] == city_name]
    feats = feats.sort_values("timestamp").reset_index(drop=True)

    window_days = int(cfg.drift.get("window_days", 14))
    threshold = float(cfg.drift.get("psi_alert_threshold", 0.25))

    registry = ModelRegistry(cfg.path("model_registry_dir"))
    bundle = registry.load_latest(city=city_name) or registry.load_latest()
    train_features: List[str] = bundle["metadata"]["features"] if bundle else []

    psi_scores: Dict[str, float] = {}
    if not feats.empty and train_features:
        cutoff = feats["timestamp"].max() - pd.Timedelta(days=window_days)
        reference = feats[feats["timestamp"] < cutoff]
        recent = feats[feats["timestamp"] >= cutoff]
        for feat in train_features:
            if feat in feats.columns:
                psi_scores[feat] = _psi(
                    reference[feat].to_numpy(dtype=float),
                    recent[feat].to_numpy(dtype=float),
                )
    top_shifted = sorted(
        (f for f, v in psi_scores.items() if not np.isnan(v)),
        key=lambda f: psi_scores[f], reverse=True,
    )[:10]

    # -- Prediction-error drift -----------------------------------------
    history = read_history(cfg.path("forecast_history_dir"), city_name)
    error_rolling = pd.DataFrame()
    error_baseline = pd.DataFrame()
    if not history.empty and not feats.empty:
        raw = fs.group("raw_observations").read()
        if "city" in raw.columns:
            raw = raw[raw["city"] == city_name]
        raw = raw[["timestamp", cfg.target]].dropna()
        raw = raw.rename(columns={cfg.target: "actual_aqi"})
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        joined = history.merge(
            raw, left_on="target_timestamp", right_on="timestamp", how="inner",
        )
        if not joined.empty:
            joined["abs_error"] = (joined["predicted_aqi"] - joined["actual_aqi"]).abs()
            joined["sq_error"] = (joined["predicted_aqi"] - joined["actual_aqi"]) ** 2
            joined = joined.sort_values("target_timestamp")
            cutoff = joined["target_timestamp"].max() - pd.Timedelta(days=window_days)
            recent = joined[joined["target_timestamp"] >= cutoff]
            baseline = joined[joined["target_timestamp"] < cutoff]
            error_rolling = recent.groupby("horizon_h").agg(
                mae=("abs_error", "mean"),
                rmse=("sq_error", lambda s: float(np.sqrt(s.mean()))),
                n=("abs_error", "size"),
            ).reset_index()
            error_baseline = baseline.groupby("horizon_h").agg(
                mae=("abs_error", "mean"),
                rmse=("sq_error", lambda s: float(np.sqrt(s.mean()))),
                n=("abs_error", "size"),
            ).reset_index()

    # -- Alert decision -------------------------------------------------
    alert_details: List[str] = []
    for f in top_shifted:
        v = psi_scores.get(f, float("nan"))
        if not np.isnan(v) and v > threshold:
            alert_details.append(f"Feature PSI drift: {f}={v:.3f} (>{threshold:.2f})")
    if not error_rolling.empty and not error_baseline.empty:
        merged = error_rolling.merge(error_baseline, on="horizon_h", suffixes=("_recent", "_base"))
        for _, row in merged.iterrows():
            if row["mae_base"] > 0 and row["mae_recent"] > 1.5 * row["mae_base"]:
                alert_details.append(
                    f"Error drift at h={row['horizon_h']}: recent MAE {row['mae_recent']:.1f} vs baseline {row['mae_base']:.1f}"
                )

    report = DriftReport(
        city=city_name,
        computed_at=pd.Timestamp.utcnow().isoformat(),
        feature_psi=psi_scores,
        top_shifted_features=top_shifted,
        error_rolling=error_rolling,
        error_baseline=error_baseline,
        alert=bool(alert_details),
        alert_details=alert_details,
    )

    # Persist report
    out = cfg.path("reports_dir") / f"drift_{city_name.replace(' ', '_')}.json"
    import json
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    _LOG.info(
        "Drift report saved (%s): alert=%s, %d shifted features",
        out, report.alert, len(top_shifted),
    )
    return report


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--city", type=str, default=None)
    args = p.parse_args()
    compute_drift(city=args.city)
