# Pearls AQI Predictor - Advanced Edition

> Predict the **US Air Quality Index (AQI)** for multiple cities over the
> next **72 hours** using a 100 % serverless MLOps stack: automated
> feature/training pipelines, a versioned model registry, uncertainty
> quantification, drift monitoring, a Streamlit dashboard **and** a
> FastAPI REST service.

```
                          Hourly (GitHub Actions)                    Daily (GitHub Actions)
                     +------------------------------+          +--------------------------+
                     |  Feature pipeline (per city) |          | Training pipeline        |
Weather + AQ APIs -> |  * fetch                     |------->  |  * rolling-origin CV     |
(Open-Meteo)         |  * clean + impute            |          |  * Optuna tuning         |
                     |  * lag / rolling / cyclic    |          |  * multi-model bake-off  |
                     |  * FUTURE-WEATHER features   |          |  * quantile P10/P50/P90  |
                     +--------------+---------------+          +------------+-------------+
                                    v                                        v
                          +-------------------+                    +-------------------+
                          |  Feature store    |  <---- reads ----- |  Model registry   |
                          |  (Parquet, city-  |                    |  (joblib +        |
                          |   partitioned)    |                    |   metadata,       |
                          |                   |                    |   per-city ptr)   |
                          +---------+---------+                    +----+---+----------+
                                    |                                   |   |
                                    v                                   v   v
                             +--------------+   +----------------+  +------------+  +----------+
                             | Streamlit v2 |   | FastAPI REST   |  | Drift      |  | Alerts   |
                             | (forecast +  |   | /forecast      |  | monitor    |  | (webhook |
                             |  model card  |   | /alerts /drift |  | (PSI +     |  |  + JSON) |
                             |  + drift)    |   | /model /cities |  |  error)    |  |          |
                             +--------------+   +----------------+  +------------+  +----------+
```

## What's new in this Advanced Edition

| Area | v1 (basic) | v2 (advanced) |
|------|------------|---------------|
| Cities | 1 | **N** (config-driven, per-city models) |
| Features | 40 | **92**: adds **future-weather forecast features** (`<var>_fh<h>`), missingness flags, wind decomposition |
| Models | Ridge / RF / GBM / LGBM / MLP / SARIMAX / naive | + **PyTorch LSTM** + **LightGBM quantile** (P10/P50/P90) |
| Tuning | none | **Optuna** with rolling-origin CV |
| Evaluation | chronological hold-out | + rolling-origin CV baseline |
| Explainability | SHAP top features | (same) |
| Alerts | console + JSON + webhook | (same) |
| Serving | Streamlit dashboard | + **FastAPI REST API** |
| Monitoring | none | **drift monitor** (feature PSI + prediction error) + **forecast history log** |
| CLI | separate scripts | one entry point: `python -m src <command>` |

## Quick start

```bash
# 1) Environment
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
# PyTorch wheels are enormous - install from PyTorch's CPU index once:
pip install "torch>=2.2,<3.0" --index-url https://download.pytorch.org/whl/cpu

# 2) Bootstrap: 365 days of history for every configured city
python -m src backfill --days 365

# 3) Train + tune every model for every city and register the winners
python -m src train

# 4) Preview the forecast for one city
python -m src predict --city Karachi

# 5a) Interactive dashboard (city switcher + uncertainty band + model card + drift)
python -m src dashboard

# 5b) OR spin up the REST API
python -m src serve   # http://127.0.0.1:8000/docs
```

## Unified CLI

```bash
python -m src backfill  [--days 365] [--city Karachi]
python -m src features  [--past-days 14] [--city Karachi]
python -m src train     [--city Karachi]
python -m src predict   [--city Karachi]
python -m src eda
python -m src drift     [--city Karachi]
python -m src alerts    [--city Karachi]
python -m src serve     [--host 0.0.0.0] [--port 8000]
python -m src dashboard
python -m src all       [--city Karachi]    # backfill -> train -> predict
```

## FastAPI endpoints

| Path | What it returns |
|------|-----------------|
| `GET /health`            | `{"status":"ok"}` |
| `GET /cities`            | List of configured cities + default |
| `GET /forecast?city=...` | Full 72h forecast + P10/P50/P90 bands |
| `GET /alerts?city=...`   | Hazardous-AQI alerts derived from the forecast |
| `GET /model?city=...`    | Model card (name, version, metrics, features) |
| `GET /drift?city=...`    | Latest drift report |

Interactive docs at http://127.0.0.1:8000/docs.

## Models

| Class | Model | Notes |
|-------|-------|-------|
| Baseline | `naive_persistence` | y_hat(t+h) = y(t) |
| Linear | `ridge` (tuned) | L2 + Optuna over alpha |
| Trees | `random_forest`, `gradient_boosting`, `lightgbm` | Multi-output regression |
| **Probabilistic** | `lightgbm_quantile` | Trains one LightGBM per (horizon, quantile) - powers the P10/P90 band |
| Neural | `mlp`, **`lstm` (PyTorch)** | LSTM uses a 48-hour lookback window |
| Statistical | `sarimax` | Seasonal ARIMA baseline |

Model selection is by **mean RMSE across horizons**. The winning point
model is registered under `latest__<city>.json`, the quantile model
under `quantile__<city>.json`. Predictions blend both so the dashboard
always shows uncertainty bands when available.

## Feature pipeline v2

Beyond the calendar/lag/rolling/change-rate/interaction features from
v1, the pipeline now also computes:

- **Missingness flags** - `<var>_isna` (0/1) for every raw variable, so
  the model can distinguish "the sensor was offline" from "the value
  really was 0".
- **Future-weather features** - for every horizon `h` and every variable
  in `variables.future_weather`, we store `<var>_fh<h>` = that
  variable's value at time `t + h`. At inference time we already know
  Open-Meteo's weather forecast for those hours, so the ML model gets
  to condition on tomorrow's wind and rain rather than pretend the
  future is unknown. This is the biggest single accuracy lever in v2 -
  test-set RMSE for Karachi dropped from **10.9 -> 6.7** just from
  adding these columns.

## Drift monitor

`python -m src drift --city Karachi` computes:

1. **PSI (Population Stability Index)** between the training tail and
   the last `drift.window_days` days for every registered feature.
2. **Prediction-error drift** by joining the forecast history log
   (`data/forecast_history/`) against the ground-truth observations and
   computing recent vs. baseline MAE / RMSE per horizon.

If any PSI exceeds `drift.psi_alert_threshold` or recent MAE is >= 1.5x
baseline, the report flags an alert. Reports land in
`reports/drift_<city>.json` and appear on the dashboard's "Drift monitor"
page.

## Windows 11 + Smart App Control note

If a fresh venv on Windows fails with
`ImportError: DLL load failed while importing interval: An Application
Control policy has blocked this file`, your `pip install` picked up
**pandas 3.0.x** or **numpy 2.5+**, whose freshly-released DLLs do not
yet have Smart App Control reputation. `requirements.txt` pins upper
bounds to avoid this. To repair an already-broken venv:

```powershell
.\env\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  "pandas>=2.0,<3.0" "pytz>=2023.3" "numpy>=1.24,<2.5" "scipy>=1.10,<1.18"
```

## Project layout

```
.
├── config.yaml                    # cities, models, tuning, drift, thresholds
├── requirements.txt
├── README.md, REPORT.md
├── src/
│   ├── __main__.py                # unified CLI
│   ├── api.py                     # FastAPI app
│   ├── config.py                  # multi-city aware
│   ├── data_fetcher.py            # Open-Meteo client (per-city)
│   ├── feature_store.py           # local Parquet feature store
│   ├── feature_pipeline.py        # v2 - multi-city + future-weather
│   ├── backfill.py                # CLI wrapper
│   ├── models.py                  # 9 model factory (+ LSTM + quantile)
│   ├── tuning.py                  # Optuna + rolling-origin CV
│   ├── training_pipeline.py       # multi-city training + registry
│   ├── model_registry.py          # per-city / per-channel pointers
│   ├── predict.py                 # forecast + intervals + history log
│   ├── forecast_history.py        # append-only forecast log per city
│   ├── drift_monitor.py           # PSI + error drift
│   ├── explain.py                 # SHAP feature importance
│   ├── alerts.py                  # hazardous-AQI alerts
│   └── utils.py
├── notebooks/eda.py
├── app/streamlit_app.py           # v2 - forecast / model card / drift pages
├── .github/workflows/
│   ├── feature_pipeline.yml       # hourly, all cities
│   └── training_pipeline.yml      # daily, all cities
├── data/
│   ├── feature_store/
│   └── forecast_history/          # rolling forecast log per city
├── models/
└── reports/                       # training_summary.json, drift_<city>.json, alerts.json, EDA
```

## Sample results (Karachi, 365-day backfill, 30-day test)

| Model | Mean RMSE | Mean MAE | Mean R² |
|-------|----------:|---------:|--------:|
| **ridge (Optuna-tuned)** | **6.69** | **4.96** | **0.523** |
| lightgbm                 | 6.73 | 5.25 | 0.464 |
| gradient_boosting        | 6.87 | 5.41 | 0.433 |
| lightgbm_quantile (P50)  | 6.91 | 5.49 | 0.430 |
| random_forest            | 7.71 | 6.01 | 0.374 |
| naive_persistence        | 7.91 | 5.90 | 0.305 |
| lstm (PyTorch)           | 8.07 | 6.45 | 0.335 |
| mlp                      | 8.95 | 7.30 | 0.067 |
| sarimax                  | 21.13 | 19.67 | -2.95 |

RMSE is **~38 % lower** than the v1 setup (10.9), thanks primarily to
future-weather features + Optuna tuning.
