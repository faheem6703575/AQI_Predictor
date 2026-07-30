"""Streamlit dashboard - v2.

Sections
--------
- Forecast page   : city switcher, current AQI, 72h forecast with P10/P90
                    band, per-horizon table, hazardous alerts, SHAP top
                    features.
- Model card page : the metadata card of the production model + all
                    challenger results.
- Drift page      : feature drift (PSI) + prediction-error drift over
                    time.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerts import evaluate_forecast_for_alerts  # noqa: E402
from src.config import load_config  # noqa: E402
from src.drift_monitor import compute_drift  # noqa: E402
from src.explain import feature_importance  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.forecast_history import read_history  # noqa: E402
from src.model_registry import ModelRegistry  # noqa: E402
from src.predict import predict_forecast  # noqa: E402
from src.utils import AQI_CATEGORIES, aqi_category  # noqa: E402


st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

CFG = load_config()


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_recent(_root: str, city: str) -> pd.DataFrame:
    fs = FeatureStore(Path(_root))
    df = fs.group("raw_observations").read()
    if not df.empty and "city" in df.columns:
        df = df[df["city"] == city]
    if not df.empty:
        df = df.sort_values("timestamp").tail(24 * 14)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def forecast_payload(city: str) -> dict:
    fc = predict_forecast(city=city, log_history=False)
    return fc.to_dict()


@st.cache_data(ttl=1800, show_spinner=False)
def importance_table(city: str) -> pd.DataFrame:
    try:
        return feature_importance(CFG.use_city(city)).head(15)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def model_metadata(city: str) -> dict | None:
    reg = ModelRegistry(CFG.path("model_registry_dir"))
    bundle = reg.load_latest(city=city)
    return bundle["metadata"] if bundle else None


@st.cache_data(ttl=900, show_spinner=False)
def drift_report(city: str) -> dict:
    return compute_drift(city=city).to_dict()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Pearls AQI")
    st.caption("Serverless 3-day AQI forecasting")
    st.markdown("---")

    city_names = [c["name"] for c in CFG.cities]
    default_ix = city_names.index(CFG.default_city_name) if CFG.default_city_name in city_names else 0
    selected_city = st.selectbox("City", city_names, index=default_ix)

    page = st.radio("Page", ["Forecast", "Model card", "Drift monitor"], index=0)

    st.markdown("---")
    meta = model_metadata(selected_city)
    if meta:
        st.markdown("**Production model**")
        st.markdown(f"- Name: `{meta['name']}`")
        st.markdown(f"- Version: `{meta['version']}`")
        st.markdown(f"- Test RMSE: `{meta['metrics'].get('mean_rmse', float('nan')):.2f}`")
    if st.button("Refresh caches"):
        st.cache_data.clear()
        st.rerun()


# ===========================================================================
# Page 1 - Forecast
# ===========================================================================
def render_forecast() -> None:
    st.title(f"Pearls AQI Predictor - {selected_city}")

    try:
        payload = forecast_payload(selected_city)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No forecast available yet: {exc}")
        st.info(
            "Bootstrap:\n\n"
            "```bash\npython -m src backfill --days 365\npython -m src train\n```"
        )
        st.stop()

    horizon_df = pd.DataFrame(payload["horizon_predictions"])
    hourly_df = pd.DataFrame(payload["hourly_curve"])
    horizon_df["timestamp"] = pd.to_datetime(horizon_df["timestamp"])
    hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])

    current_obs = load_recent(str(CFG.path("feature_store_dir")), selected_city)
    current_aqi = float(current_obs[CFG.target].iloc[-1]) if not current_obs.empty else float("nan")
    current_cat = aqi_category(current_aqi)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current AQI", f"{current_aqi:.0f}", current_cat["category"])
    for col, h in zip((c2, c3, c4), (24, 48, 72)):
        row = horizon_df.loc[horizon_df["horizon_h"] == h]
        col.metric(f"In {h}h", f"{row['aqi'].iloc[0]:.0f}" if not row.empty else "-")

    st.markdown(
        f"<div style='padding:0.6rem;border-radius:8px;background:{current_cat['color']};"
        f"color:black;font-weight:600'>"
        f"Current category: {current_cat['category']} - {current_cat['message']}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- Chart -----------------------------------------------------------
    st.subheader("3-day AQI forecast")
    fig = go.Figure()

    if not current_obs.empty:
        hist = current_obs.tail(24 * 7)
        fig.add_trace(go.Scatter(
            x=hist["timestamp"], y=hist[CFG.target],
            name="Observed (last 7 days)",
            line=dict(color="#1f77b4", width=2),
        ))

    if "q10" in hourly_df.columns and "q90" in hourly_df.columns:
        # P10-P90 uncertainty band
        fig.add_trace(go.Scatter(
            x=hourly_df["timestamp"], y=hourly_df["q90"],
            name="P90", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=hourly_df["timestamp"], y=hourly_df["q10"],
            name="P10-P90 band", fill="tonexty",
            fillcolor="rgba(214,39,40,0.15)", line=dict(width=0),
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=hourly_df["timestamp"], y=hourly_df["aqi"],
        name="Forecast (median)",
        line=dict(color="#d62728", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=horizon_df["timestamp"], y=horizon_df["aqi"],
        name="Horizon predictions", mode="markers+text",
        text=[f"+{h}h" for h in horizon_df["horizon_h"]],
        textposition="top center",
        marker=dict(size=10, color=horizon_df["color"], line=dict(width=1, color="black")),
    ))
    for lo, hi, _name, color in AQI_CATEGORIES:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.07, line_width=0)
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Time (UTC)", yaxis_title="AQI",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Alerts ----------------------------------------------------------
    st.subheader("Hazardous-AQI alerts")
    alerts = evaluate_forecast_for_alerts(horizon_df, cfg=CFG.use_city(selected_city))
    if alerts:
        for a in alerts:
            st.error(f"[{a.level.upper()}] {a.message}")
    else:
        st.success("No hazardous AQI levels forecast in the next 72 hours.")

    # --- Table + Importance ---------------------------------------------
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Per-horizon predictions")
        table = horizon_df.copy()
        table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%MZ")
        display_cols = ["horizon_h", "timestamp", "aqi", "category"]
        for q in ("q10", "q50", "q90"):
            if q in table.columns:
                display_cols.append(q)
        rename = {"horizon_h": "Hours ahead", "aqi": "AQI (P50)", "category": "Category"}
        st.dataframe(
            table[display_cols].rename(columns=rename),
            hide_index=True, use_container_width=True,
        )
    with right:
        st.subheader("Top features (SHAP)")
        imp = importance_table(selected_city)
        if imp.empty:
            st.info("Feature importance not available yet.")
        else:
            st.bar_chart(imp.set_index("feature")["importance"])

    st.caption(
        f"Forecast issued at {payload['issued_at']} using "
        f"{payload['model_name']}@{payload['model_version']}. "
        "Hourly features + daily training run via GitHub Actions."
    )


# ===========================================================================
# Page 2 - Model card
# ===========================================================================
def render_model_card() -> None:
    st.title("Model card")
    meta = model_metadata(selected_city)
    if meta is None:
        st.warning("No model registered yet.")
        return

    st.subheader(f"{meta['name']} @ {meta['version']}")
    st.write(f"City: **{selected_city}**  |  Target: **{meta['target']}**")
    st.write(f"Features used: **{len(meta['features'])}**  |  Horizons: **{meta['horizons']}**")

    st.subheader("Test-set metrics (last 30 days)")
    metrics = meta.get("metrics", {}).get("per_horizon", {})
    if metrics:
        mdf = pd.DataFrame(metrics).T.rename_axis("Horizon").reset_index()
        st.dataframe(mdf, hide_index=True, use_container_width=True)
    st.write("Mean RMSE across horizons: ", meta.get("metrics", {}).get("mean_rmse"))

    st.subheader("Challenger models")
    all_results = meta.get("extra", {}).get("all_results", {})
    if all_results:
        rows = []
        for name, r in all_results.items():
            if isinstance(r, dict) and "mean_rmse" in r:
                rows.append({
                    "model": name,
                    "mean_rmse": r["mean_rmse"],
                    "mean_mae": r["mean_mae"],
                    "mean_r2": r["mean_r2"],
                })
        if rows:
            df = pd.DataFrame(rows).sort_values("mean_rmse").reset_index(drop=True)
            st.dataframe(df, hide_index=True, use_container_width=True)

    cv = meta.get("extra", {}).get("cv", {})
    if cv:
        st.subheader("Rolling-origin CV RMSE")
        st.dataframe(pd.DataFrame(cv).T, use_container_width=True)

    tuning = meta.get("extra", {}).get("best_hyperparams", {})
    if tuning:
        st.subheader("Best tuned hyperparameters")
        st.json(tuning)


# ===========================================================================
# Page 3 - Drift monitor
# ===========================================================================
def render_drift() -> None:
    st.title("Drift monitor")
    try:
        report = drift_report(selected_city)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Drift computation failed: {exc}")
        return

    if report["alert"]:
        st.error(f"Drift alerts ({len(report['alert_details'])})")
        for d in report["alert_details"]:
            st.write("- ", d)
    else:
        st.success("No drift alerts. Feature & error distributions look stable.")

    st.subheader("Top shifted features (PSI)")
    psi = report.get("feature_psi", {})
    if psi:
        rows = [(f, psi[f]) for f in report["top_shifted_features"] if f in psi]
        df = pd.DataFrame(rows, columns=["feature", "psi"]).dropna()
        if not df.empty:
            st.bar_chart(df.set_index("feature")["psi"])

    history = read_history(CFG.path("forecast_history_dir"), selected_city)
    if not history.empty:
        st.subheader(f"Forecast log ({len(history):,} rows)")
        st.dataframe(history.tail(200), hide_index=True, use_container_width=True)

    err_rolling = pd.DataFrame(report.get("error_rolling") or [])
    err_baseline = pd.DataFrame(report.get("error_baseline") or [])
    if not err_rolling.empty:
        st.subheader("Error by horizon")
        if not err_baseline.empty and "horizon_h" in err_baseline.columns and "horizon_h" in err_rolling.columns:
            merged = err_rolling.merge(
                err_baseline, on="horizon_h", suffixes=("_recent", "_base"), how="outer"
            )
            st.dataframe(merged, hide_index=True, use_container_width=True)
        else:
            st.info("No baseline error data yet for this city — showing recent errors only.")
            st.dataframe(err_rolling, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Forecast":
    render_forecast()
elif page == "Model card":
    render_model_card()
elif page == "Drift monitor":
    render_drift()
