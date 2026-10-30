"""Where intermediate and processed datasets live, plus the download manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fxml.config import Config


def bars_dir(cfg: Config, symbol: str) -> Path:
    return cfg.paths.interim / f"bars{cfg.bar_minutes}" / symbol


def manifest_path(cfg: Config) -> Path:
    return cfg.paths.interim / f"bars{cfg.bar_minutes}" / "manifest.json"


def read_manifest(cfg: Config) -> dict:
    path = manifest_path(cfg)
    return json.loads(path.read_text()) if path.exists() else {}


def write_manifest(cfg: Config, manifest: dict) -> None:
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    tmp.replace(path)


def load_bars(cfg: Config, symbol: str) -> pd.DataFrame:
    """All stored intraday bars for ``symbol``, sorted by time."""
    files = sorted(bars_dir(cfg, symbol).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no bars for {symbol}; run scripts/download_data.py first")
    bars = pd.concat(pd.read_parquet(f) for f in files).sort_index()
    return bars[~bars.index.duplicated(keep="last")]
