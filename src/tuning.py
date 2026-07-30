"""Optuna hyperparameter tuning for the top ML models.

Uses rolling-origin cross-validation on the training set so we tune for
time-series generalisation rather than random-shuffle accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

try:
    import optuna

    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:  # pragma: no cover
    HAS_OPTUNA = False

from .models import make_model
from .utils import get_logger

_LOG = get_logger("tuning")


def _rolling_origin_splits(n: int, n_folds: int, min_train: int) -> List[tuple]:
    """Yield (train_idx, val_idx) tuples for a rolling-origin CV.

    Fold ``k`` uses ``min_train + k*step`` rows for training and the next
    ``step`` rows for validation.
    """
    if n_folds < 1 or n <= min_train:
        return [(np.arange(0, int(n * 0.8)), np.arange(int(n * 0.8), n))]
    step = max(1, (n - min_train) // n_folds)
    folds = []
    for k in range(n_folds):
        train_end = min_train + k * step
        val_end = min(train_end + step, n)
        if val_end - train_end < 2 or train_end >= n:
            break
        folds.append((np.arange(0, train_end), np.arange(train_end, val_end)))
    return folds or [(np.arange(0, int(n * 0.8)), np.arange(int(n * 0.8), n))]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _search_space(name: str, trial) -> Dict[str, Any]:
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 22),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
    if name == "gradient_boosting":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if name == "mlp":
        depth = trial.suggest_int("depth", 1, 3)
        width = trial.suggest_int("width", 32, 256, step=32)
        return {
            "hidden_layer_sizes": tuple(width for _ in range(depth)),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 400, step=50),
        }
    return {}


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    horizons: List[int],
    *,
    n_trials: int = 20,
    n_folds: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Return the best hyperparameter dict for ``name`` on ``(X, y)``.

    Falls back to an empty dict if Optuna isn't installed.
    """
    if not HAS_OPTUNA:
        _LOG.info("Optuna not installed; skipping tuning for %s", name)
        return {}

    min_train = max(200, int(len(X) * 0.6))
    folds = _rolling_origin_splits(len(X), n_folds=n_folds, min_train=min_train)

    def objective(trial: "optuna.Trial") -> float:
        hp = _search_space(name, trial)
        scores: List[float] = []
        for tr_idx, va_idx in folds:
            spec = make_model(name, horizons, random_state=random_state, hyperparams=hp)
            X_tr = X.iloc[tr_idx].copy()
            X_va = X.iloc[va_idx].copy()
            X_tr.attrs["target_col"] = X.attrs.get("target_col", "us_aqi")
            X_va.attrs["target_col"] = X.attrs.get("target_col", "us_aqi")
            spec.estimator.fit(X_tr, y[tr_idx])
            pred = spec.estimator.predict(X_va)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            scores.append(_rmse(y[va_idx], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    _LOG.info(
        "Tuned %s: best_rmse=%.3f params=%s",
        name, study.best_value, study.best_params,
    )

    # Reconstruct the fit-compatible hyperparam dict (may include grouped
    # keys like hidden_layer_sizes which live outside the raw search space).
    best = _search_space(name, study.best_trial)
    return best
