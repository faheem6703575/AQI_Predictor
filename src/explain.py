"""Feature-importance explanations.

Uses SHAP when the registered model is tree- or linear-based, and falls
back to permutation importance for other estimators.

Returns a tidy ``pandas.DataFrame`` with columns ``feature`` and
``importance`` (mean absolute SHAP value or permutation drop).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import Config, load_config
from .feature_store import FeatureStore
from .model_registry import ModelRegistry
from .utils import get_logger

_LOG = get_logger("explain")


def _resolve_estimator(model):
    """Drill into Pipelines / MultiOutputRegressors to a base estimator."""
    inner = model
    if hasattr(inner, "named_steps"):
        inner = inner.named_steps.get("model", inner)
    if hasattr(inner, "estimators_") and inner.estimators_:
        inner = inner.estimators_[0]
    return inner


def feature_importance(cfg: Optional[Config] = None, sample_size: int = 500) -> pd.DataFrame:
    cfg = cfg or load_config()
    registry = ModelRegistry(cfg.path("model_registry_dir"))
    bundle = registry.load_latest()
    if bundle is None:
        raise RuntimeError("No model in registry")
    model = bundle["model"]
    meta = bundle["metadata"]
    features = meta["features"]

    fs = FeatureStore(cfg.path("feature_store_dir"))
    df = fs.group("engineered_features").read()
    target_cols = [c for c in df.columns if c.startswith("target_h")]
    feat_df = df.dropna(subset=features + target_cols)
    if feat_df.empty:
        raise RuntimeError("Not enough complete rows to explain")
    sample = feat_df.tail(sample_size)[features].reset_index(drop=True)
    sample.attrs["target_col"] = meta["target"]

    # ------------------------------------------------------------------
    # Try SHAP
    # ------------------------------------------------------------------
    try:
        import shap

        inner = _resolve_estimator(model)

        # For Pipelines wrapping a scaler, build a transformed sample so the
        # SHAP explainer sees the same input the base estimator was fit on.
        if hasattr(model, "named_steps") and "scaler" in model.named_steps:
            scaled = model.named_steps["scaler"].transform(sample)
            sample_for_shap = pd.DataFrame(scaled, columns=features)
        else:
            sample_for_shap = sample

        if hasattr(inner, "feature_importances_"):
            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(sample_for_shap)
        elif hasattr(inner, "coef_"):
            # Multi-output linear regressors expose 2D coef_; SHAP's
            # LinearExplainer wants a single-output estimator, so we use
            # KernelExplainer on the multi-output prediction averaged over
            # horizons. To keep it fast we use a small background sample.
            bg = shap.sample(sample_for_shap, min(50, len(sample_for_shap)), random_state=0)

            def _avg_predict(x: np.ndarray) -> np.ndarray:
                if hasattr(model, "named_steps") and "scaler" in model.named_steps:
                    inner_pred = model.named_steps["model"].predict(x)
                else:
                    inner_pred = model.predict(pd.DataFrame(x, columns=features))
                if inner_pred.ndim > 1:
                    return inner_pred.mean(axis=1)
                return inner_pred

            explainer = shap.KernelExplainer(_avg_predict, bg)
            shap_values = explainer.shap_values(
                sample_for_shap.sample(min(80, len(sample_for_shap)), random_state=1),
                nsamples=100,
                silent=True,
            )
        else:
            raise RuntimeError("No suitable explainer")

        if isinstance(shap_values, list):
            shap_arr = np.mean([np.abs(v) for v in shap_values], axis=0)
        else:
            shap_arr = np.abs(shap_values)
        # Make sure we collapse to per-feature mean importance even if the
        # underlying explainer returned (samples, features, outputs).
        while shap_arr.ndim > 2:
            shap_arr = shap_arr.mean(axis=-1)
        importance = shap_arr.mean(axis=0)
        out = pd.DataFrame({"feature": features, "importance": importance})
        out["source"] = "shap"
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    except Exception as exc:  # noqa: BLE001
        _LOG.info("SHAP path failed (%s); falling back to permutation importance", exc)

    # ------------------------------------------------------------------
    # Fallback: permutation importance on the first horizon
    # ------------------------------------------------------------------
    from sklearn.inspection import permutation_importance

    first_target = target_cols[0]
    y = feat_df.tail(sample_size)[first_target].to_numpy()
    base_pred = model.predict(sample)
    if base_pred.ndim > 1:
        base_pred = base_pred[:, 0]
    rmses = []
    rng = np.random.default_rng(42)
    for col in features:
        permuted = sample.copy()
        permuted[col] = rng.permutation(permuted[col].to_numpy())
        p = model.predict(permuted)
        if p.ndim > 1:
            p = p[:, 0]
        rmses.append(float(np.sqrt(np.mean((p - y) ** 2)) - np.sqrt(np.mean((base_pred - y) ** 2))))
    out = pd.DataFrame({"feature": features, "importance": rmses})
    out["source"] = "permutation"
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
