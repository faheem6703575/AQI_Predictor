"""Model factory (advanced).

All estimators expose scikit-learn style ``fit(X, y)`` / ``predict(X)``
and return ``(n_samples, n_horizons)`` arrays so the rest of the pipeline
is horizon-agnostic.

Model catalogue
---------------
Point forecasters:
    naive_persistence, ridge, random_forest, gradient_boosting,
    lightgbm, mlp, sarimax, lstm

Probabilistic forecaster:
    lightgbm_quantile - trains one LightGBM per (horizon, quantile) with the
                        quantile-regression objective. Predictions come back
                        as a dict {q: (n_samples, n_horizons)}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_SARIMAX = True
except Exception:  # pragma: no cover
    HAS_SARIMAX = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


# ===========================================================================
# Custom estimators
# ===========================================================================
class NaivePersistence(BaseEstimator, RegressorMixin):
    def __init__(self, target_col: str = "us_aqi") -> None:
        self.target_col = target_col
        self._n_horizons = 1

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "NaivePersistence":
        self._n_horizons = y.shape[1] if y.ndim > 1 else 1
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.target_col in X.columns:
            base = X[self.target_col].to_numpy().reshape(-1, 1)
        else:
            base = np.zeros((len(X), 1))
        return np.repeat(base, self._n_horizons, axis=1)


class SarimaxMultiHorizon(BaseEstimator, RegressorMixin):
    def __init__(self, horizons: List[int], order=(2, 1, 1), seasonal_order=(0, 1, 1, 24)):
        self.horizons = horizons
        self.order = order
        self.seasonal_order = seasonal_order
        self._results = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SarimaxMultiHorizon":
        if not HAS_SARIMAX:
            raise RuntimeError("statsmodels not installed")
        target_col = X.attrs.get("target_col", "us_aqi")
        series = X[target_col].to_numpy().astype(float)
        series = series[-min(len(series), 24 * 60):]
        model = SARIMAX(
            series, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        self._results = model.fit(disp=False, maxiter=50)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._results is None:
            raise RuntimeError("SARIMAX not fitted")
        max_h = max(self.horizons)
        forecast = self._results.get_forecast(steps=max_h).predicted_mean
        n = len(X)
        out = np.zeros((n, len(self.horizons)))
        for i in range(n):
            for j, h in enumerate(self.horizons):
                out[i, j] = forecast[h - 1] if h - 1 < len(forecast) else forecast[-1]
        return out


class LightGBMQuantile(BaseEstimator, RegressorMixin):
    """LightGBM quantile regression for (horizon, quantile) pairs.

    ``predict`` returns the median (0.5 quantile) so it slots into the
    normal training-pipeline evaluation, and exposes ``predict_quantiles``
    for the P10/P50/P90 bands.
    """

    def __init__(
        self,
        horizons: List[int],
        quantiles: List[float] = (0.1, 0.5, 0.9),
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        random_state: int = 42,
    ) -> None:
        self.horizons = list(horizons)
        self.quantiles = list(quantiles)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self._models: Dict[float, List[Any]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LightGBMQuantile":
        if not HAS_LGBM:
            raise RuntimeError("lightgbm not installed")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._models = {}
        for q in self.quantiles:
            per_h: List[Any] = []
            for j in range(y.shape[1]):
                mdl = LGBMRegressor(
                    objective="quantile", alpha=q,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves,
                    random_state=self.random_state, verbose=-1,
                )
                mdl.fit(X, y[:, j])
                per_h.append(mdl)
            self._models[q] = per_h
        return self

    def _predict_q(self, X: pd.DataFrame, q: float) -> np.ndarray:
        cols = [m.predict(X) for m in self._models[q]]
        return np.column_stack(cols)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        median_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
        return self._predict_q(X, median_q)

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        return {q: self._predict_q(X, q) for q in self.quantiles}


class TorchLSTM(BaseEstimator, RegressorMixin):
    """A small PyTorch LSTM that maps a lookback window of features to a
    multi-horizon AQI forecast.

    We accept ``pandas.DataFrame`` inputs so it plugs into the same code
    path as the sklearn models. Sequences are built on the fly from the
    ``us_aqi`` column plus its lag features (which are already time-shifted
    versions of the target - we treat the last ``seq_len`` rows of X as
    the input sequence).
    """

    def __init__(
        self,
        horizons: List[int],
        seq_len: int = 48,
        hidden_size: int = 64,
        num_layers: int = 1,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.horizons = list(horizons)
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

    # --- helpers ---------------------------------------------------------
    def _make_sequences(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in X.columns if X[c].dtype != object]
        arr = X[cols].astype(np.float32).to_numpy()
        n, d = arr.shape
        if n < self.seq_len:
            pad = np.repeat(arr[:1], self.seq_len - n, axis=0)
            arr = np.concatenate([pad, arr], axis=0)
            n = arr.shape[0]
        seqs = np.zeros((n, self.seq_len, d), dtype=np.float32)
        for i in range(n):
            lo = max(0, i - self.seq_len + 1)
            block = arr[lo : i + 1]
            if block.shape[0] < self.seq_len:
                pad = np.repeat(block[:1], self.seq_len - block.shape[0], axis=0)
                block = np.concatenate([pad, block], axis=0)
            seqs[i] = block
        return seqs

    # --- fit / predict ---------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TorchLSTM":
        if not HAS_TORCH:
            raise RuntimeError("torch not installed")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        torch.manual_seed(self.random_state)

        cols = [c for c in X.columns if X[c].dtype != object]
        self._feature_cols = cols
        arr = X[cols].astype(np.float32).to_numpy()
        # Feature scaling stats (avoid divide-by-zero)
        self._mean = arr.mean(axis=0)
        self._std = arr.std(axis=0) + 1e-6
        arr_scaled = (arr - self._mean) / self._std
        X_scaled = pd.DataFrame(arr_scaled, columns=cols)
        seqs = self._make_sequences(X_scaled)
        y_mean = float(y.mean())
        y_std = float(y.std() + 1e-6)
        self._y_mean, self._y_std = y_mean, y_std
        y_scaled = (y - y_mean) / y_std

        device = torch.device("cpu")
        self._device = device
        self._model = _LSTMNet(
            input_size=arr.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=y.shape[1],
        ).to(device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.SmoothL1Loss()

        X_t = torch.tensor(seqs, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_scaled, dtype=torch.float32, device=device)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True
        )
        self._model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optim.zero_grad()
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not HAS_TORCH:
            raise RuntimeError("torch not installed")
        cols = getattr(self, "_feature_cols", None) or list(X.columns)
        arr = X[cols].astype(np.float32).to_numpy()
        arr_scaled = (arr - self._mean) / self._std
        seqs = self._make_sequences(pd.DataFrame(arr_scaled, columns=cols))
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(seqs, dtype=torch.float32, device=self._device)
            pred = self._model(X_t).cpu().numpy()
        return pred * self._y_std + self._y_mean


if HAS_TORCH:

    class _LSTMNet(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last)


# ===========================================================================
# Factory
# ===========================================================================
@dataclass
class ModelSpec:
    name: str
    estimator: Any
    handles_multioutput: bool = True
    is_naive: bool = False
    is_probabilistic: bool = False


def make_model(
    name: str,
    horizons: List[int],
    *,
    random_state: int = 42,
    quantiles: Optional[List[float]] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> ModelSpec:
    """Build a fresh, unfitted estimator by name.

    ``hyperparams`` overrides the defaults; used by the Optuna tuner.
    """
    hp = dict(hyperparams or {})
    name = name.lower()

    if name == "naive_persistence":
        return ModelSpec(name, NaivePersistence(), is_naive=True)

    if name == "ridge":
        alpha = hp.get("alpha", 1.0)
        return ModelSpec(
            name,
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha, random_state=random_state)),
            ]),
        )

    if name == "random_forest":
        return ModelSpec(
            name,
            RandomForestRegressor(
                n_estimators=hp.get("n_estimators", 200),
                max_depth=hp.get("max_depth", 14),
                min_samples_leaf=hp.get("min_samples_leaf", 2),
                n_jobs=-1, random_state=random_state,
            ),
        )

    if name == "gradient_boosting":
        return ModelSpec(
            name,
            MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_depth=hp.get("max_depth", 8),
                    learning_rate=hp.get("learning_rate", 0.05),
                    max_iter=hp.get("max_iter", 300),
                    random_state=random_state,
                )
            ),
        )

    if name == "lightgbm":
        if not HAS_LGBM:
            raise RuntimeError("lightgbm not installed")
        return ModelSpec(
            name,
            MultiOutputRegressor(
                LGBMRegressor(
                    n_estimators=hp.get("n_estimators", 400),
                    learning_rate=hp.get("learning_rate", 0.05),
                    num_leaves=hp.get("num_leaves", 63),
                    subsample=hp.get("subsample", 0.9),
                    colsample_bytree=hp.get("colsample_bytree", 0.9),
                    random_state=random_state, verbose=-1,
                )
            ),
        )

    if name == "lightgbm_quantile":
        if not HAS_LGBM:
            raise RuntimeError("lightgbm not installed")
        q = list(quantiles) if quantiles else [0.1, 0.5, 0.9]
        return ModelSpec(
            name,
            LightGBMQuantile(
                horizons=horizons,
                quantiles=q,
                n_estimators=hp.get("n_estimators", 300),
                learning_rate=hp.get("learning_rate", 0.05),
                num_leaves=hp.get("num_leaves", 63),
                random_state=random_state,
            ),
            is_probabilistic=True,
        )

    if name == "mlp":
        return ModelSpec(
            name,
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=hp.get("hidden_layer_sizes", (128, 64)),
                    activation="relu", solver="adam",
                    learning_rate_init=hp.get("lr", 1e-3),
                    batch_size=64, max_iter=hp.get("max_iter", 200),
                    early_stopping=True, random_state=random_state,
                )),
            ]),
        )

    if name == "lstm":
        if not HAS_TORCH:
            raise RuntimeError("torch not installed")
        return ModelSpec(
            name,
            TorchLSTM(
                horizons=horizons,
                seq_len=hp.get("seq_len", 48),
                hidden_size=hp.get("hidden_size", 64),
                num_layers=hp.get("num_layers", 1),
                epochs=hp.get("epochs", 20),
                batch_size=hp.get("batch_size", 64),
                lr=hp.get("lr", 1e-3),
                random_state=random_state,
            ),
        )

    if name == "sarimax":
        return ModelSpec(name, SarimaxMultiHorizon(horizons=horizons))

    raise ValueError(f"Unknown model: {name}")


def available_models() -> Dict[str, bool]:
    return {
        "naive_persistence": True,
        "ridge": True,
        "random_forest": True,
        "gradient_boosting": True,
        "lightgbm": HAS_LGBM,
        "lightgbm_quantile": HAS_LGBM,
        "mlp": True,
        "lstm": HAS_TORCH,
        "sarimax": HAS_SARIMAX,
    }
