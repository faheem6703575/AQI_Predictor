"""Training pipeline (advanced).

Advances over v1
----------------
* **Per-city training** - trains an independent model per city and
  registers each under its own name (``ridge_<city>``).
* **Rolling-origin cross-validation** for robust time-series evaluation.
* **Optuna hyperparameter tuning** for a configurable subset of models.
* **Quantile-regression model** kept alongside the point-forecast winner
  so the predict path can surface prediction intervals.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import Config, load_config
from .feature_store import FeatureStore
from .model_registry import ModelRegistry
from .models import ModelSpec, available_models, make_model
from .tuning import _rolling_origin_splits, tune_model
from .utils import get_logger

_LOG = get_logger("training")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_frame(cfg: Config) -> pd.DataFrame:
    fs = FeatureStore(cfg.path("feature_store_dir"))
    df = fs.group("engineered_features").read()
    if df.empty:
        raise RuntimeError(
            "Feature store is empty. Run `python -m src.backfill --days 365` first."
        )
    if "city" in df.columns:
        df = df[df["city"] == cfg.city["name"]].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(
            f"No rows for city={cfg.city['name']} in the feature store."
        )
    return df


def build_feature_matrix(
    df: pd.DataFrame, cfg: Config
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    horizons = cfg.horizons
    target_cols = [f"target_h{h}" for h in horizons]
    drop_cols = {"timestamp", "city", *target_cols}

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    work = df.dropna(subset=target_cols).copy()
    work = work.dropna(subset=feature_cols).reset_index(drop=True)

    X = work[feature_cols].copy()
    X.attrs["target_col"] = cfg.target
    y = work[target_cols].to_numpy()
    return X, y, feature_cols, target_cols


def chronological_split(
    X: pd.DataFrame,
    y: np.ndarray,
    df: pd.DataFrame,
    test_size_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    timestamps = df.loc[X.index, "timestamp"] if "timestamp" in df.columns else None
    if timestamps is None:
        raise RuntimeError("timestamp column missing")
    cutoff = timestamps.max() - pd.Timedelta(days=test_size_days)
    train_mask = timestamps <= cutoff
    test_mask = ~train_mask
    return (
        X.loc[train_mask].reset_index(drop=True),
        X.loc[test_mask].reset_index(drop=True),
        y[train_mask.to_numpy()],
        y[test_mask.to_numpy()],
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, horizons: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_horizon": {}}
    rmses, maes, r2s = [], [], []
    for i, h in enumerate(horizons):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rmse = _safe_rmse(yt, yp)
        mae = float(mean_absolute_error(yt, yp))
        try:
            r2 = float(r2_score(yt, yp))
        except Exception:  # noqa: BLE001
            r2 = float("nan")
        out["per_horizon"][f"h{h}"] = {"rmse": rmse, "mae": mae, "r2": r2}
        rmses.append(rmse); maes.append(mae); r2s.append(r2)
    out["mean_rmse"] = float(np.mean(rmses))
    out["mean_mae"] = float(np.mean(maes))
    out["mean_r2"] = float(np.nanmean(r2s))
    return out


def cross_validated_rmse(
    spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, n_folds: int,
) -> Dict[str, float]:
    """Rolling-origin CV RMSE using ``n_folds`` splits on ``X``."""
    min_train = max(200, int(len(X) * 0.6))
    folds = _rolling_origin_splits(len(X), n_folds=n_folds, min_train=min_train)
    rmses = []
    for tr_idx, va_idx in folds:
        spec2 = make_model(spec.name, [], random_state=42) if spec.is_naive \
            else make_model(spec.name, list(range(y.shape[1])), random_state=42)
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        X_tr.attrs["target_col"] = X.attrs.get("target_col", "us_aqi")
        X_va.attrs["target_col"] = X.attrs.get("target_col", "us_aqi")
        spec2.estimator.fit(X_tr, y[tr_idx])
        pred = spec2.estimator.predict(X_va)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        rmses.append(_safe_rmse(y[va_idx], pred))
    return {"cv_mean_rmse": float(np.mean(rmses)), "cv_std_rmse": float(np.std(rmses)), "cv_folds": len(rmses)}


# ---------------------------------------------------------------------------
# Training entry-points
# ---------------------------------------------------------------------------
def _train_one(
    spec: ModelSpec, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame,
    y_te: np.ndarray, horizons: List[int],
) -> Tuple[Any, Dict[str, Any]]:
    _LOG.info("Training %s ...", spec.name)
    spec.estimator.fit(X_tr, y_tr)
    y_pred = spec.estimator.predict(X_te)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.shape[1] != y_te.shape[1]:
        if y_pred.shape[1] == 1:
            y_pred = np.repeat(y_pred, y_te.shape[1], axis=1)
    metrics = evaluate(y_te, y_pred, horizons)
    _LOG.info(
        "%s: mean_rmse=%.3f mean_mae=%.3f mean_r2=%.3f",
        spec.name, metrics["mean_rmse"], metrics["mean_mae"], metrics["mean_r2"],
    )
    return spec.estimator, metrics


def _train_city(cfg_city: Config) -> Dict[str, Any]:
    target = cfg_city.target
    horizons = cfg_city.horizons
    city_name = cfg_city.city["name"]

    df = load_training_frame(cfg_city)
    X, y, feature_cols, _ = build_feature_matrix(df, cfg_city)
    if len(X) < 200:
        raise RuntimeError(
            f"[{city_name}] Only {len(X)} training rows after feature build."
        )

    df_for_split = df.loc[X.index].reset_index(drop=True)
    X_tr, X_te, y_tr, y_te = chronological_split(
        X, y, df_for_split,
        test_size_days=int(cfg_city.training.get("test_size_days", 30)),
    )
    _LOG.info("[%s] train=%d test=%d features=%d", city_name, len(X_tr), len(X_te), X_tr.shape[1])

    avail = available_models()
    tuning_cfg = cfg_city.tuning
    tune_targets = set(tuning_cfg.get("models_to_tune", [])) if tuning_cfg.get("enabled") else set()

    summary: Dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "city": city_name,
        "target": target,
        "horizons": horizons,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "feature_count": len(feature_cols),
        "results": {},
        "cv": {},
        "best_hyperparams": {},
    }
    best_name: Optional[str] = None
    best_score = float("inf")
    best_estimator = None
    quantile_estimator = None
    quantile_metadata: Dict[str, Any] = {}

    for model_name in cfg_city.training.get("models", []):
        if not avail.get(model_name, False):
            _LOG.warning("[%s] skipping %s (unavailable)", city_name, model_name)
            continue
        try:
            hp = {}
            if model_name in tune_targets:
                _LOG.info("[%s] tuning %s", city_name, model_name)
                hp = tune_model(
                    model_name, X_tr, y_tr, horizons,
                    n_trials=int(tuning_cfg.get("n_trials", 20)),
                    n_folds=int(cfg_city.training.get("cv_folds", 3)),
                    random_state=int(cfg_city.training.get("random_state", 42)),
                )
                summary["best_hyperparams"][model_name] = hp

            spec = make_model(
                model_name, horizons,
                random_state=int(cfg_city.training.get("random_state", 42)),
                quantiles=cfg_city.quantiles,
                hyperparams=hp,
            )
            estimator, metrics = _train_one(spec, X_tr.copy(), y_tr, X_te.copy(), y_te, horizons)
            summary["results"][model_name] = metrics

            # Rolling-origin CV score (skipped for the slower/naive models)
            if model_name not in {"naive_persistence", "sarimax", "lstm"}:
                try:
                    cv_scores = cross_validated_rmse(
                        spec, X_tr, y_tr,
                        n_folds=int(cfg_city.training.get("cv_folds", 3)),
                    )
                    summary["cv"][model_name] = cv_scores
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("[%s] CV failed for %s: %s", city_name, model_name, exc)

            if model_name == "lightgbm_quantile":
                quantile_estimator = estimator
                quantile_metadata = {
                    "quantiles": list(cfg_city.quantiles),
                    "metrics": metrics,
                }
            else:
                if metrics["mean_rmse"] < best_score:
                    best_score = metrics["mean_rmse"]
                    best_name = model_name
                    best_estimator = estimator
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("[%s] %s failed: %s", city_name, model_name, exc)
            summary["results"][model_name] = {"error": str(exc)}

    if best_name is None:
        raise RuntimeError(f"[{city_name}] no models trained successfully")

    summary["best_model"] = best_name
    summary["best_mean_rmse"] = best_score

    # Register the point-forecast model
    registry = ModelRegistry(cfg_city.path("model_registry_dir"))
    registry.register(
        best_estimator,
        name=f"{best_name}__{city_name}",
        metrics=summary["results"][best_name],
        features=feature_cols,
        horizons=horizons,
        target=target,
        extra={
            "city": city_name,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "all_results": summary["results"],
            "cv": summary["cv"],
            "best_hyperparams": summary["best_hyperparams"],
        },
        promote_latest_key=f"latest__{city_name}.json",
    )

    # Register the quantile model separately (for prediction intervals)
    if quantile_estimator is not None:
        registry.register(
            quantile_estimator,
            name=f"lightgbm_quantile__{city_name}",
            metrics=quantile_metadata["metrics"],
            features=feature_cols,
            horizons=horizons,
            target=target,
            extra={
                "city": city_name,
                "quantiles": quantile_metadata["quantiles"],
            },
            promote_latest_key=f"quantile__{city_name}.json",
        )

    return summary


def run_training(
    cfg: Optional[Config] = None,
    *,
    cities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    city_names = cities or [c["name"] for c in cfg.cities]
    combined: Dict[str, Any] = {"per_city": {}, "trained_at": datetime.now(timezone.utc).isoformat()}
    for name in city_names:
        try:
            combined["per_city"][name] = _train_city(cfg.use_city(name))
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("[%s] training failed: %s", name, exc)
            combined["per_city"][name] = {"error": str(exc)}
    reports_dir = cfg.path("reports_dir")
    (reports_dir / "training_summary.json").write_text(json.dumps(combined, indent=2))
    return combined


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--city", type=str, default=None)
    args = p.parse_args()
    run_training(cities=[args.city] if args.city else None)
