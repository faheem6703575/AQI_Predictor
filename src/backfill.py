"""Convenience wrapper to run a historical backfill.

Usage:
    python -m src.backfill --days 365
"""

from __future__ import annotations

import argparse

from .feature_pipeline import run_backfill


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features")
    parser.add_argument("--days", type=int, default=None, help="Number of past days")
    parser.add_argument("--chunk-days", type=int, default=90)
    args = parser.parse_args()
    run_backfill(backfill_days=args.days, chunk_days=args.chunk_days)


if __name__ == "__main__":
    main()
