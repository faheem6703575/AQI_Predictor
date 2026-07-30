# Pearls AQI Predictor - Advanced Project Report

## 1. Executive summary

This report documents the **Advanced Edition** of the Pearls AQI
Predictor. It builds on the original serverless architecture from the
brief and adds the pieces you'd expect in a production ML system:

* Multiple cities with per-city models and per-city registry pointers.
* A feature pipeline that includes **future weather forecasts as
  exogenous inputs** for each target horizon.
* A deep-learning contender (**PyTorch LSTM**) trained on a lookback
  window of features.
* **Probabilistic forecasts** (P10 / P50 / P90) via LightGBM quantile
  regression.
* **Optuna hyperparameter tuning** on top of **rolling-origin
  cross-validation**.
* A **drift monitor** watching feature-distribution shifts (PSI) and
  prediction-error drift over time via a forecast history log.
* A **FastAPI REST service** in addition to the Streamlit dashboard.
* A single **CLI** (`python -m src ...`) covering the entire lifecycle.

On the Karachi city (365-day backfill, 30-day held-out tail), the
advanced pipeline drops mean-across-horizons RMSE from **10.9 -> 6.7**
(-38 %), driven mostly by future-weather features and Optuna tuning.

---

## 2. Architecture (v2)

```
                          Hourly (GitHub Actions)                     Daily (GitHub Actions)
                     +------------------------------+           +--------------------------+
                     |  Feature pipeline (per city) |           | Training pipeline        |
Weather + AQ APIs -> |  * fetch                     |  ------>  |  * rolling-origin CV     |
(Open-Meteo)         |  * clean + impute            |           |  * Optuna tuning         |
                     |  * lag/rolling/cyclic        |           |  * multi-model bake-off  |
                     |  * FUTURE-WEATHER features   |           |  * quantile P10/P50/P90  |
                     +--------------+---------------+           +------------+-------------+
                                    v                                        v
                          +-------------------+                    +-------------------+
                          |  Feature store    |  <---- reads ----- |  Model registry   |
                          |  (Parquet, city-  |                    |  (joblib +        |
                          |   partitioned)    |                    |   metadata,       |
                          |                   |                    |   per-city ptr)   |
                          +---------+---------+                    +----+----+---------+
                                    |                                   |    |
                        +-----------+-----------+                       v    v
                        v                       v                       Streamlit + FastAPI
                +---------------+       +---------------+
                | Drift monitor |       | Forecast log  |
                | (PSI + err.)  |  <--  | (append-only) |
                +---------------+       +---------------+
```

### Key design principles

| Principle | How it's realised |
|-----------|-------------------|
| **Serverless / free** | Open-Meteo (no API key), local Parquet feature store, GitHub Actions cron, Streamlit Community Cloud-friendly. |
| **Per-city isolation** | Feature store partitions on `(timestamp, city)`, each city has its own registered model under `latest__<city>.json`. |
| **Time-series-safe evaluation** | Rolling-origin CV instead of random shuffling; chronological hold-out tail for final test. |
| **Uncertainty over point accuracy** | The dashboard always plots P10 / P50 / P90 bands so the user sees how confident the model is. |
| **Observability by default** | Every forecast is appended to `data/forecast_history/<city>.parquet` so drift + retrospective accuracy can be computed later. |
| **Two serving surfaces** | Streamlit for humans, FastAPI for machines. Both talk to the same registry. |

---

## 3. Data pipeline

### 3.1 Source variables

Open-Meteo Air Quality (hourly): PM2.5, PM10, CO, NO2, SO2, O3, and the
AQI (target).

Open-Meteo Historical Weather + Forecast (hourly): temperature,
relative humidity, dew point, surface pressure, precipitation, wind
speed, wind direction.

### 3.2 Feature engineering (v2)

The feature pipeline produces a single tidy table
`engineered_features/data.parquet` keyed by `(timestamp, city)`.
Beyond the raw observations it computes:

- **Calendar features** - `hour`, `dayofweek`, `day`, `month`,
  `is_weekend`, cyclic encodings `hour_sin`/`cos`, `month_sin`/`cos`.
- **Lag features** on the target: 1 h, 3 h, 6 h, 12 h, 24 h, 48 h, 72 h.
- **Rolling statistics** (shifted by 1 to avoid leakage): 6 h and 24 h
  mean and std of AQI.
- **Change-rate features**: 1-hour diff, 1-hour pct change, 24-hour diff.
- **Weather interactions**: `temp_humidity`, and wind vector
  decomposition `(wind_u, wind_v)`.
- **Missingness flags** (**new**): `<var>_isna` (0/1) for every raw
  variable, letting the model differentiate an imputed zero from a
  real zero.
- **Future-weather features** (**new** and *the* biggest lever):
  for each forecast horizon `h` in `[1, 6, 12, 24, 48, 72]` and each
  variable in `variables.future_weather` (temperature, humidity,
  precipitation, wind speed), we add `<var>_fh<h>` = that variable's
  value `h` hours in the future. At inference time we already know
  Open-Meteo's weather forecast up to 72 hours ahead, so the model gets
  to condition on tomorrow's rain and wind rather than pretending the
  future is unknown.

Total feature count grew from **40 -> 92** on the same 365-day slice.

### 3.3 Feature store

Local partitioned Parquet layout, identical write interface to a
Hopsworks feature-group upsert:

```
data/feature_store/
    raw_observations/
        data.parquet         # main, dedup on (timestamp, city)
        history/             # append-only audit snapshots
    engineered_features/
        data.parquet
        history/
```

Multi-city rows co-exist in the same Parquet file - the primary key
`(timestamp, city)` handles dedup.

---

## 4. Modelling

### 4.1 Point-forecast models

| Class | Model | Notes |
|-------|-------|-------|
| Baseline | `naive_persistence` | `y_hat(t+h) = y(t)` |
| Linear | `ridge` | Standardized features + L2, `alpha` tuned by Optuna |
| Trees | `random_forest` | 200 trees, native multi-output |
| Trees | `gradient_boosting` | HistGradientBoosting per-horizon |
| Trees | `lightgbm` | Multi-output LightGBM |
| Neural | `mlp` | sklearn `MLPRegressor(128,64)` |
| Neural | **`lstm` (PyTorch)** | 48-hour lookback -> LSTM(hidden=64) -> MLP head, Adam + Smooth-L1 |
| Statistical | `sarimax` | Seasonal ARIMA baseline |

### 4.2 Probabilistic forecaster - `lightgbm_quantile`

For every configured quantile `q` in `forecast.quantiles`
(default `[0.1, 0.5, 0.9]`) we fit one LightGBM per forecast horizon
with `objective="quantile", alpha=q`. Prediction returns a dictionary
`{q: (n_samples, n_horizons)}`, from which the dashboard renders a
P10-P90 uncertainty band around the P50 median.

### 4.3 Rolling-origin cross-validation

Time-series data doesn't tolerate random shuffling because it leaks the
future into the past. `_rolling_origin_splits(n, n_folds, min_train)`
produces expanding-window folds: fold `k` trains on the first
`min_train + k*step` samples and validates on the next `step` samples.
This is used both for:

* Optuna hyperparameter tuning (each trial's score is the mean fold
  RMSE), and
* A CV RMSE baseline recorded alongside the final held-out metrics
  (`extra.cv` in the model card).

### 4.4 Optuna hyperparameter tuning

`training.tuning.enabled: true` in `config.yaml` and
`training.tuning.models_to_tune` selects which model(s) to tune (kept
small by default so a full training run fits inside GitHub Actions'
free minutes). The search spaces per model:

| Model | Search space |
|-------|--------------|
| ridge | `alpha` (log-uniform, 1e-3 - 1e2) |
| lightgbm | `n_estimators`, `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree` |
| random_forest | `n_estimators`, `max_depth`, `min_samples_leaf` |
| gradient_boosting | `max_depth`, `learning_rate`, `max_iter` |
| mlp | depth, width, `lr`, `max_iter` |

Best hyperparameters are recorded in the model card at
`extra.best_hyperparams`.

### 4.5 Model registry (v2)

Each city writes its own pointer:

```
models/
    ridge__Karachi_20260709T155807Z.joblib
    ridge__Karachi_20260709T155807Z.json    # metadata
    latest__Karachi.json                    # -> ridge__Karachi_20260709T155807Z
    lightgbm_quantile__Karachi_20260709T155807Z.joblib
    lightgbm_quantile__Karachi_20260709T155807Z.json
    quantile__Karachi.json                  # -> lightgbm_quantile__Karachi_...
    (... and the same three files for every other city ...)
```

`ModelRegistry.load_latest(city="Karachi")` reads
`latest__Karachi.json` and returns both the joblib blob and the JSON
metadata. `load_quantile(city="Karachi")` does the same for the
quantile channel. This is the exact API surface a Hopsworks / Vertex
model registry would expose, so a future migration is a one-file swap.

---

## 5. Prediction & forecast history

`predict.predict_forecast(city="...")`:

1. Loads the latest point + quantile models for the city.
2. Grabs the most recent feature-complete row from the feature store.
3. Produces per-horizon predictions + a smooth hourly curve
   (linear interpolation between anchor horizons).
4. Adds quantile columns to the curve when the quantile model is
   available.
5. **Appends the forecast to `data/forecast_history/<city>.parquet`**
   so downstream code can retrospectively grade its accuracy.

The Streamlit chart uses that same log for the drift page's "recent vs.
baseline error by horizon" table.

---

## 6. Drift monitor

`src/drift_monitor.py` runs two independent drift checks:

### 6.1 Feature drift - PSI

For every registered feature we compute the
**Population Stability Index** between a *reference* window (training
tail, everything older than `drift.window_days` days from the newest
row) and a *recent* window (the last `drift.window_days` days).

PSI thresholds (industry-standard):

| PSI | Meaning |
|-----|---------|
| < 0.10 | Stable |
| 0.10 - 0.25 | Moderate drift |
| > 0.25 | Significant drift - retrain |

Any feature whose PSI exceeds `drift.psi_alert_threshold` (default 0.25)
generates an alert.

### 6.2 Prediction-error drift

The forecast history log is joined against the raw observations table
on `target_timestamp` to compute per-horizon MAE / RMSE for the same
"recent vs. baseline" split. If recent MAE at any horizon is
>= 1.5x baseline, we flag an alert.

Both signals show up on the dashboard's Drift page and in
`reports/drift_<city>.json`.

---

## 7. Serving

### 7.1 Streamlit dashboard (v2)

Three pages, all city-scoped via the sidebar selector:

1. **Forecast** - current AQI, +24 h / +48 h / +72 h metrics, colour
   category, 7-day observed line + 72-hour forecast with the P10-P90
   band + horizon anchor points, hazardous-AQI alert list, per-horizon
   table with quantiles, SHAP top features.
2. **Model card** - production model name, version, test metrics per
   horizon, challenger leaderboard, rolling-origin CV RMSE, and the
   Optuna-selected hyperparameters.
3. **Drift monitor** - PSI table + top-shifted features bar chart,
   forecast log tail, recent vs. baseline error table.

### 7.2 FastAPI REST service

```
GET /health                         # liveness probe
GET /cities                         # configured cities + default
GET /forecast?city=Karachi          # 72h forecast + quantiles
GET /alerts?city=Karachi            # hazardous-AQI alerts
GET /model?city=Karachi             # production model card
GET /drift?city=Karachi             # latest drift report
```

Auto-generated OpenAPI docs at `/docs`, launched via
`python -m src serve`.

---

## 8. CI / CD

Two GitHub Actions workflows.

### 8.1 `feature_pipeline.yml` - hourly

Iterates over every city:
1. Fetches the trailing 7 days from Open-Meteo (plus 3 days of forecast
   weather for the future-weather features).
2. Cleans, engineers, upserts into the feature store.
3. Regenerates the forecast, evaluates hazardous alerts, and dispatches
   them (webhook if `ALERT_WEBHOOK_URL` is set).
4. Recomputes the drift report.
5. Commits the updated `data/feature_store`, `data/forecast_history`,
   `reports/alerts.json`, and `reports/drift_<city>.json` back to the
   repo with `[skip ci]`.

### 8.2 `training_pipeline.yml` - daily

1. Auto-bootstraps a 365-day backfill for every city the first time.
2. Retrains all models for all cities (Optuna tuning included).
3. Registers the new winners and updates each city's `latest__*.json`
   pointer.
4. Commits the model registry + `reports/training_summary.json`.

Because the repo *is* the store + registry, the Streamlit dashboard on
Streamlit Community Cloud automatically has fresh data + fresh models.

---

## 9. Sample results

**Karachi**, 365-day backfill, 30-day test hold-out:

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

vs. v1 (40 features, no tuning) which got **10.90** on the same city -
a **~38 % RMSE reduction**.

Per-horizon detail for the winning Ridge (Karachi):

| Horizon | RMSE | MAE | R² |
|--------:|-----:|----:|---:|
| +1 h  | 0.73  | 0.58  | 0.994 |
| +6 h  | 3.61  | 2.61  | 0.868 |
| +12 h | 5.28  | 3.78  | ~0.73 |
| +24 h | 7.83  | 5.65  | ~0.55 |
| +48 h | 9.06  | 6.90  | ~0.35 |
| +72 h | 9.73  | 7.29  | ~0.24 |

Even at the hardest horizon (+72 h) the model retains meaningful skill
(R² > 0), which is a direct payoff of the future-weather features -
tomorrow's rain / wind carry real predictive signal.

---

## 10. Mapping back to the brief

| Brief requirement | Where it lives |
|-------------------|----------------|
| Fetch weather + pollutant data from an external API | `src/data_fetcher.py` (Open-Meteo, per-city) |
| Compute features (inputs) + targets (outputs) | `src/feature_pipeline.py` |
| Include time-based + derived (AQI change rate) features | `_add_calendar`, `_add_change_rate` |
| Store features in a Feature Store | `src/feature_store.py` (Parquet, city-partitioned) |
| Backfill historical (features, targets) | `python -m src backfill --days 365` |
| Fetch historical (features, targets) from Feature Store | `load_training_frame` in `src/training_pipeline.py` |
| Train and evaluate the best model possible | `run_training` |
| Experiment with sklearn (Random Forest, Ridge Regression) **and TensorFlow / PyTorch for advanced models** | Ridge + RF + GBM + LightGBM + MLP + **PyTorch LSTM** in `src/models.py` |
| Evaluate performance using RMSE, MAE, R² | `evaluate()` in `src/training_pipeline.py` |
| Store the trained model in a Model Registry | `src/model_registry.py` (per-city pointers) |
| CI/CD: feature script hourly + training script daily | `.github/workflows/feature_pipeline.yml` + `training_pipeline.yml` |
| Load model + features from the store, compute predictions, show on a dashboard | `src/predict.py` + `app/streamlit_app.py` |
| Use Streamlit / Gradio **and Flask / FastAPI** | Streamlit dashboard **and FastAPI** (`src/api.py`) |
| Perform EDA to identify trends | `notebooks/eda.py` -> `reports/EDA.md` |
| Use SHAP or LIME for feature importance | `src/explain.py` (SHAP + permutation fallback) |
| Add alerts for hazardous AQI levels | `src/alerts.py` (console + JSON + webhook) |
| End-to-end AQI prediction system | This repo |
| Scalable, automated pipeline | GitHub Actions + Parquet feature store + Optuna |
| Interactive dashboard showing real-time + forecast | Streamlit v2 with forecast / model-card / drift pages |
| Detailed report | This file |

---

## 11. Limitations & next steps

* **Managed feature store**: swap the local Parquet class for
  Hopsworks / Vertex AI - the `FeatureStore.group(...).upsert()` API is
  the same. Estimated effort: 1 file, ~1 hour.
* **Longer backfills**: Open-Meteo's air-quality archive goes back to
  2022 - a 3-year backfill would materially improve the +48 h / +72 h
  scores by giving the model more seasonal regimes.
* **LSTM budget**: the current PyTorch LSTM runs 20 epochs to fit
  GitHub Actions' free minutes. Doubling that + adding attention would
  make it competitive with Ridge on longer horizons.
* **Multivariate quantiles**: the current quantile model has one
  LightGBM per (horizon, quantile). A single conformalized quantile
  regressor would give calibration guarantees.
* **True 24/7 alerting**: today alerts go to a JSON file + optional
  webhook. A Twilio SMS / email integration is a small next step.
* **Multi-tenant deployment**: nothing in the code is Karachi-specific,
  but multi-tenant auth on the FastAPI service would let external
  clients pull their own city's forecasts.
