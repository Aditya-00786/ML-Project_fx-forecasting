"""Daily series from FRED (Federal Reserve Bank of St. Louis), downloaded as CSV with no API key."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def download_series(series_id: str, dest_dir: Path, session: requests.Session) -> Path:
    response = session.get(CSV_URL.format(series_id=series_id), timeout=60)
    response.raise_for_status()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{series_id}.csv"
    path.write_text(response.text)
    return path


def parse_series(text: str) -> pd.Series:
    """Parse a FRED CSV (a date column and one value column). Missing values become NaN."""
    raw = pd.read_csv(io.StringIO(text))
    date_col, value_col = raw.columns[:2]
    values = pd.to_numeric(raw[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(raw[date_col]), name=value_col)
    series.index.name = "date"
    return series


def load_series(raw_dir: Path, mapping: dict[str, str]) -> pd.DataFrame:
    """Combine saved series into one frame with columns named by ``mapping``."""
    columns = {
        name: parse_series((raw_dir / f"{series_id}.csv").read_text())
        for series_id, name in mapping.items()
    }
    return pd.DataFrame(columns).sort_index()
