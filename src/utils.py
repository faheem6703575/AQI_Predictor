"""Shared utilities: logging, AQI category helpers, retryable HTTP."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import requests


def get_logger(name: str = "aqi") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# AQI categorization
# ---------------------------------------------------------------------------
AQI_CATEGORIES = [
    (0, 50, "Good", "#00E400"),
    (51, 100, "Moderate", "#FFFF00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF7E00"),
    (151, 200, "Unhealthy", "#FF0000"),
    (201, 300, "Very Unhealthy", "#8F3F97"),
    (301, 500, "Hazardous", "#7E0023"),
]


def aqi_category(value: Optional[float]) -> Dict[str, Any]:
    """Map a US-AQI value to a category, color, and health message."""
    if value is None or (isinstance(value, float) and value != value):
        return {"category": "Unknown", "color": "#999999", "message": "No data"}
    for lo, hi, name, color in AQI_CATEGORIES:
        if lo <= value <= hi:
            return {
                "category": name,
                "color": color,
                "range": (lo, hi),
                "message": _HEALTH_MESSAGES[name],
            }
    if value > 500:
        return {
            "category": "Beyond Hazardous",
            "color": "#4C0013",
            "range": (501, float("inf")),
            "message": _HEALTH_MESSAGES["Hazardous"],
        }
    return {"category": "Unknown", "color": "#999999", "message": "No data"}


_HEALTH_MESSAGES = {
    "Good": "Air quality is satisfactory and air pollution poses little or no risk.",
    "Moderate": "Acceptable; sensitive individuals may experience minor effects.",
    "Unhealthy for Sensitive Groups": "Sensitive groups should limit outdoor exertion.",
    "Unhealthy": "Everyone may begin to experience health effects. Avoid prolonged exertion.",
    "Very Unhealthy": "Health alert: everyone may experience more serious health effects.",
    "Hazardous": "Health warning of emergency conditions. The entire population is more likely to be affected.",
}


# ---------------------------------------------------------------------------
# Retryable HTTP GET
# ---------------------------------------------------------------------------
def http_get_json(
    url: str,
    params: Dict[str, Any],
    *,
    timeout: int = 30,
    retries: int = 3,
    backoff: float = 5.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or _LOG
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("HTTP attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    assert last_exc is not None
    raise last_exc


def time_it(label: str, fn: Callable[[], Any], logger: Optional[logging.Logger] = None) -> Any:
    log = logger or _LOG
    start = time.time()
    out = fn()
    log.info("%s took %.2fs", label, time.time() - start)
    return out
