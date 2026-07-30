"""FastAPI REST wrapper around the trained model.

Endpoints
---------
GET /health                 - liveness/readiness
GET /cities                 - list of configured cities
GET /forecast?city=Karachi  - 72-hour forecast + intervals for a city
GET /alerts?city=Karachi    - hazardous-AQI alerts for the current forecast
GET /model?city=Karachi     - production model card (name, version, metrics)
GET /drift?city=Karachi     - latest drift report

Run with:
    python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
or via the CLI:
    python -m src serve
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query

from .alerts import evaluate_forecast_for_alerts
from .config import load_config
from .drift_monitor import compute_drift
from .model_registry import ModelRegistry
from .predict import predict_forecast

app = FastAPI(
    title="Pearls AQI Predictor API",
    version="2.0.0",
    description="Serverless multi-city 3-day AQI forecasting service.",
)


def _cfg():
    return load_config()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/cities")
def cities() -> Dict[str, Any]:
    cfg = _cfg()
    return {"default": cfg.default_city_name, "cities": cfg.cities}


@app.get("/forecast")
def forecast(city: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    cfg = _cfg()
    target_city = city or cfg.default_city_name
    try:
        fc = predict_forecast(cfg=cfg, city=target_city)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return fc.to_dict()


@app.get("/alerts")
def alerts(city: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    cfg = _cfg()
    target_city = city or cfg.default_city_name
    fc = predict_forecast(cfg=cfg, city=target_city, log_history=False)
    alerts = evaluate_forecast_for_alerts(fc.horizon_predictions, cfg=cfg.use_city(target_city))
    return {
        "city": target_city,
        "count": len(alerts),
        "alerts": [a.__dict__ for a in alerts],
    }


@app.get("/model")
def model_card(city: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    cfg = _cfg()
    target_city = city or cfg.default_city_name
    registry = ModelRegistry(cfg.path("model_registry_dir"))
    bundle = registry.load_latest(city=target_city)
    if bundle is None:
        raise HTTPException(status_code=404, detail="No model registered for that city")
    return bundle["metadata"]


@app.get("/drift")
def drift(city: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    cfg = _cfg()
    target_city = city or cfg.default_city_name
    report = compute_drift(cfg=cfg, city=target_city)
    return report.to_dict()
