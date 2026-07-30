"""Unified command-line interface.

Usage::

    python -m src backfill [--days 365] [--city Karachi]
    python -m src features [--past-days 14] [--city Karachi]
    python -m src train    [--city Karachi]
    python -m src predict  [--city Karachi]
    python -m src eda
    python -m src drift    [--city Karachi]
    python -m src serve    [--host 0.0.0.0] [--port 8000]
    python -m src dashboard
    python -m src alerts   [--city Karachi]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .utils import get_logger

_LOG = get_logger("cli")


def _cmd_backfill(args: argparse.Namespace) -> None:
    from .feature_pipeline import run_backfill

    run_backfill(
        backfill_days=args.days,
        chunk_days=args.chunk_days,
        cities=[args.city] if args.city else None,
    )


def _cmd_features(args: argparse.Namespace) -> None:
    from .feature_pipeline import run_feature_pipeline

    run_feature_pipeline(
        past_days=args.past_days,
        cities=[args.city] if args.city else None,
    )


def _cmd_train(args: argparse.Namespace) -> None:
    from .training_pipeline import run_training

    run_training(cities=[args.city] if args.city else None)


def _cmd_predict(args: argparse.Namespace) -> None:
    from .predict import predict_forecast

    fc = predict_forecast(city=args.city)
    print(fc.horizon_predictions.to_string(index=False))


def _cmd_eda(_args: argparse.Namespace) -> None:
    from notebooks.eda import run_eda

    run_eda()


def _cmd_drift(args: argparse.Namespace) -> None:
    from .drift_monitor import compute_drift

    report = compute_drift(city=args.city)
    print(f"City: {report.city}, alert: {report.alert}")
    for d in report.alert_details:
        print(" -", d)


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)


def _cmd_dashboard(_args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(root / "app" / "streamlit_app.py")],
        check=False,
    )


def _cmd_alerts(args: argparse.Namespace) -> None:
    from .alerts import dispatch, evaluate_forecast_for_alerts
    from .predict import predict_forecast

    cfg = load_config()
    if args.city:
        cfg = cfg.use_city(args.city)
    fc = predict_forecast(cfg=cfg, city=args.city, log_history=False)
    alerts = evaluate_forecast_for_alerts(fc.horizon_predictions, cfg=cfg)
    dispatch(alerts, cfg)
    print(f"Dispatched {len(alerts)} alert(s) for {cfg.city['name']}")


def _cmd_all(args: argparse.Namespace) -> None:
    _cmd_backfill(args)
    _cmd_train(args)
    _cmd_predict(args)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src")
    sub = p.add_subparsers(dest="command", required=True)

    def add_city(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--city", type=str, default=None)

    b = sub.add_parser("backfill"); add_city(b)
    b.add_argument("--days", type=int, default=None)
    b.add_argument("--chunk-days", type=int, default=90)
    b.set_defaults(func=_cmd_backfill)

    f = sub.add_parser("features"); add_city(f)
    f.add_argument("--past-days", type=int, default=14)
    f.set_defaults(func=_cmd_features)

    t = sub.add_parser("train"); add_city(t)
    t.set_defaults(func=_cmd_train)

    pr = sub.add_parser("predict"); add_city(pr)
    pr.set_defaults(func=_cmd_predict)

    e = sub.add_parser("eda")
    e.set_defaults(func=_cmd_eda)

    d = sub.add_parser("drift"); add_city(d)
    d.set_defaults(func=_cmd_drift)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)

    da = sub.add_parser("dashboard")
    da.set_defaults(func=_cmd_dashboard)

    al = sub.add_parser("alerts"); add_city(al)
    al.set_defaults(func=_cmd_alerts)

    a = sub.add_parser("all", help="backfill -> train -> predict"); add_city(a)
    a.add_argument("--days", type=int, default=None)
    a.add_argument("--chunk-days", type=int, default=90)
    a.set_defaults(func=_cmd_all)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
