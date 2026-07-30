"""A minimal local Feature Store.

This is intentionally lightweight so the project stays 100% serverless and
zero-cost: we persist features in partitioned Parquet files on disk. The
public API mirrors what you would get from a managed feature store (e.g.
Hopsworks, Vertex AI), so swapping in a cloud backend later is a one-file
change.

Layout::

    data/feature_store/
        <feature_group>/
            data.parquet        # latest, deduplicated by timestamp + city
            history/            # append-only raw snapshots (audit log)

Each feature group stores hourly rows for a single city, identified by an
ISO-8601 UTC ``timestamp`` column.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .utils import get_logger

_LOG = get_logger("feature_store")


@dataclass
class FeatureGroup:
    name: str
    root: Path
    primary_keys: List[str]

    @property
    def main_path(self) -> Path:
        return self.root / "data.parquet"

    @property
    def history_dir(self) -> Path:
        return self.root / "history"

    # ---- read ------------------------------------------------------------
    def read(self) -> pd.DataFrame:
        if not self.main_path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self.main_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ---- write -----------------------------------------------------------
    def upsert(self, df: pd.DataFrame) -> int:
        """Insert/update rows; dedupes on ``primary_keys``. Returns rows after."""
        if df is None or df.empty:
            return self._row_count()
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        self.root.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # 1. Audit snapshot
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        df.to_parquet(self.history_dir / f"snapshot_{stamp}.parquet", index=False)

        # 2. Merge into main
        existing = self.read()
        combined = (
            pd.concat([existing, df], ignore_index=True)
            if not existing.empty
            else df
        )
        combined = combined.drop_duplicates(subset=self.primary_keys, keep="last")
        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        combined.to_parquet(self.main_path, index=False)

        _LOG.info(
            "Upserted %d rows into feature_group=%s (total=%d)",
            len(df),
            self.name,
            len(combined),
        )
        return len(combined)

    # ---- misc ------------------------------------------------------------
    def _row_count(self) -> int:
        return 0 if not self.main_path.exists() else len(self.read())

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)


class FeatureStore:
    """Thin facade so callers don't talk to Parquet paths directly."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def group(
        self,
        name: str,
        primary_keys: Optional[Iterable[str]] = None,
    ) -> FeatureGroup:
        pks = list(primary_keys) if primary_keys else ["timestamp", "city"]
        return FeatureGroup(name=name, root=self.root / name, primary_keys=pks)
