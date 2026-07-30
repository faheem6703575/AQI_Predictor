"""Exploratory Data Analysis for the Pearls AQI Predictor.

Runs *after* a backfill is available and writes a set of figures + a small
HTML/Markdown summary into ``reports/figures``.

    python -m notebooks.eda

Sections
--------
1. Coverage & missingness
2. AQI distribution and category breakdown
3. Time-series trend (daily mean, weekly mean)
4. Diurnal & weekly cycles
5. Pollutant correlations
6. AQI vs. weather scatter
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import load_config
from src.feature_store import FeatureStore
from src.utils import AQI_CATEGORIES, aqi_category, get_logger

_LOG = get_logger("eda")
sns.set_theme(style="whitegrid", context="talk")


def _categorize(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cats = df[target].apply(aqi_category)
    df = df.copy()
    df["category"] = cats.apply(lambda d: d["category"])
    return df


def run_eda() -> Path:
    cfg = load_config()
    fs = FeatureStore(cfg.path("feature_store_dir"))
    df = fs.group("raw_observations").read()
    if df.empty:
        raise SystemExit("Feature store is empty. Run a backfill first.")
    out_dir = Path(cfg.path("reports_dir")) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = cfg.target
    df = _categorize(df, target)

    summary_lines: list[str] = []
    summary_lines.append(f"# EDA report - {cfg.city['name']}")
    summary_lines.append("")
    summary_lines.append(f"- Rows: **{len(df):,}**")
    summary_lines.append(
        f"- Date range: **{df['timestamp'].min()} -> {df['timestamp'].max()}**"
    )
    summary_lines.append(f"- Mean {target}: **{df[target].mean():.1f}**")
    summary_lines.append(f"- Max {target}: **{df[target].max():.1f}**")
    summary_lines.append("")

    # --- 1. Coverage & missingness -----------------------------------------
    miss = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    miss.head(20).plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Fraction missing")
    ax.set_title("Top columns by missing-rate")
    fig.tight_layout()
    fig.savefig(out_dir / "01_missingness.png", dpi=120)
    plt.close(fig)

    # --- 2. AQI distribution + category breakdown --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[target].dropna(), bins=60, ax=axes[0], color="darkorange")
    axes[0].set_title(f"Distribution of {target}")
    counts = df["category"].value_counts().reindex(
        [c[2] for c in AQI_CATEGORIES] + ["Beyond Hazardous"], fill_value=0,
    )
    palette = {c[2]: c[3] for c in AQI_CATEGORIES}
    palette["Beyond Hazardous"] = "#4C0013"
    counts.plot.bar(ax=axes[1], color=[palette[c] for c in counts.index])
    axes[1].set_title("Hours by AQI category")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "02_distribution_categories.png", dpi=120)
    plt.close(fig)

    # --- 3. Time-series trend ----------------------------------------------
    ts = df.set_index("timestamp")[target].resample("D").mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ts.plot(ax=ax, color="firebrick", linewidth=1.4)
    ts.rolling(7).mean().plot(ax=ax, color="black", linewidth=1, label="7-day rolling")
    ax.set_title(f"Daily mean {target}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "03_trend.png", dpi=120)
    plt.close(fig)

    # --- 4. Diurnal & weekly cycles ----------------------------------------
    hourly = df.groupby(df["timestamp"].dt.hour)[target].mean()
    dow = df.groupby(df["timestamp"].dt.dayofweek)[target].mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    hourly.plot(ax=axes[0], marker="o", color="navy")
    axes[0].set_title("Diurnal cycle (mean AQI by hour)")
    axes[0].set_xlabel("Hour of day")
    dow.plot(ax=axes[1], marker="o", color="teal")
    axes[1].set_title("Weekly cycle (mean AQI by day-of-week)")
    axes[1].set_xlabel("Day of week (0=Mon)")
    fig.tight_layout()
    fig.savefig(out_dir / "04_cycles.png", dpi=120)
    plt.close(fig)

    # --- 5. Pollutant correlations -----------------------------------------
    pollutant_cols = [
        c for c in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone", target]
        if c in df.columns
    ]
    corr = df[pollutant_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Pollutant correlations")
    fig.tight_layout()
    fig.savefig(out_dir / "05_pollutant_correlations.png", dpi=120)
    plt.close(fig)

    # --- 6. AQI vs weather scatter -----------------------------------------
    weather_cols = [
        c for c in ["temperature_2m", "relative_humidity_2m",
                    "wind_speed_10m", "precipitation"]
        if c in df.columns
    ]
    if weather_cols:
        sample = df.sample(min(len(df), 5000), random_state=42)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, col in zip(axes.flat, weather_cols):
            ax.scatter(sample[col], sample[target], s=6, alpha=0.4)
            ax.set_xlabel(col)
            ax.set_ylabel(target)
            ax.set_title(f"{target} vs {col}")
        fig.tight_layout()
        fig.savefig(out_dir / "06_weather_vs_aqi.png", dpi=120)
        plt.close(fig)

    summary_lines.append("## Figures")
    for f in sorted(out_dir.glob("*.png")):
        summary_lines.append(f"![{f.stem}](figures/{f.name})")

    report_md = Path(cfg.path("reports_dir")) / "EDA.md"
    report_md.write_text("\n".join(summary_lines), encoding="utf-8")
    _LOG.info("EDA written to %s", report_md)
    return report_md


if __name__ == "__main__":
    run_eda()
