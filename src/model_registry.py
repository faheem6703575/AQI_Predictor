"""Local Model Registry (advanced).

Each registered model gets a versioned ``.joblib`` blob plus a JSON
metadata sidecar. In addition to a global ``latest.json`` pointer, we
support **named** pointers (e.g. ``latest__Karachi.json``,
``quantile__Karachi.json``) so different cities and different model
"channels" (point forecast vs. quantile) can co-exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from .utils import get_logger

_LOG = get_logger("model_registry")


@dataclass
class ModelMetadata:
    name: str
    version: str
    created_at: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    horizons: List[int] = field(default_factory=list)
    target: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def register(
        self,
        model: Any,
        *,
        name: str,
        metrics: Dict[str, Any],
        features: List[str],
        horizons: List[int],
        target: str,
        extra: Optional[Dict[str, Any]] = None,
        promote_latest: bool = True,
        promote_latest_key: Optional[str] = None,
    ) -> ModelMetadata:
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_path = self.root / f"{name}_{version}.joblib"
        meta_path = self.root / f"{name}_{version}.json"

        joblib.dump(model, model_path)
        meta = ModelMetadata(
            name=name, version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics, features=features, horizons=horizons,
            target=target, extra=extra or {},
        )
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2))

        if promote_latest:
            pointer_name = promote_latest_key or "latest.json"
            (self.root / pointer_name).write_text(
                json.dumps({"name": name, "version": version}, indent=2)
            )
        _LOG.info("Registered %s@%s", name, version)
        return meta

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def load_pointer(self, pointer_name: str = "latest.json") -> Optional[Dict[str, Any]]:
        ptr = self.root / pointer_name
        if not ptr.exists():
            return None
        pointer = json.loads(ptr.read_text())
        return self.load(pointer["name"], pointer["version"])

    def load_latest(self, city: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if city:
            named = self.load_pointer(f"latest__{city}.json")
            if named is not None:
                return named
        return self.load_pointer("latest.json")

    def load_quantile(self, city: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if city:
            return self.load_pointer(f"quantile__{city}.json")
        return self.load_pointer("quantile.json")

    def load(self, name: str, version: str) -> Dict[str, Any]:
        model_path = self.root / f"{name}_{version}.joblib"
        meta_path = self.root / f"{name}_{version}.json"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        return {
            "model": joblib.load(model_path),
            "metadata": json.loads(meta_path.read_text()),
        }

    def list_models(self) -> List[Dict[str, Any]]:
        out = []
        for meta_path in sorted(self.root.glob("*.json")):
            if meta_path.name.startswith("latest") or meta_path.name.startswith("quantile"):
                continue
            out.append(json.loads(meta_path.read_text()))
        return out
