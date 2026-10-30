"""Central bank policy rates from the BIS bulk download (flat CSV inside a ZIP).

The uncompressed file is large (about 0.5 GB) because every row repeats long labels, so it is
streamed row by row straight from the ZIP.
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BULK_URL = "https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip"


def download(dest_dir: Path, session: requests.Session) -> Path:
    response = session.get(BULK_URL, timeout=300)
    response.raise_for_status()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / BULK_URL.rsplit("/", 1)[-1]
    path.write_bytes(response.content)
    return path


def _code(value: str) -> str:
    """BIS cells look like ``"D: Daily"``; keep the code before the colon."""
    return value.split(":", 1)[0].strip()


def read_observations(zip_path: Path, areas: set[str], since: date) -> pd.DataFrame:
    """Daily and monthly observations for ``areas`` from ``since`` onwards, in long format."""
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as handle:
            reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8"))
            header = next(reader)
            col = {_code(h): i for i, h in enumerate(header)}
            f, a, t, v = col["FREQ"], col["REF_AREA"], col["TIME_PERIOD"], col["OBS_VALUE"]
            for row in reader:
                freq, area = _code(row[f]), _code(row[a])
                if freq in ("D", "M") and area in areas and row[t] >= since.isoformat()[:7]:
                    rows.append((freq, area, row[t], row[v]))
    obs = pd.DataFrame(rows, columns=["freq", "area", "period", "value"])
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    return obs


def policy_rates(obs: pd.DataFrame, areas: dict[str, str], start: date, end: date) -> pd.DataFrame:
    """One column per currency on every calendar day from ``start`` to ``end``, forward filled.

    Daily observations are used where they exist. Before a country's daily series begins,
    its end-of-month values fill the gap. If a series still starts after ``start`` (Japan's
    begins on 9 March 2006, during the Bank of Japan's zero-rate policy), the first published
    value is carried back to ``start``.
    """
    calendar = pd.date_range(start, end, freq="D", name="date")
    out = {}
    for area, currency in areas.items():
        mine = obs[obs["area"] == area].dropna(subset=["value"])
        daily = mine[mine["freq"] == "D"]
        daily = pd.Series(daily["value"].to_numpy(), index=pd.DatetimeIndex(daily["period"]))
        monthly = mine[mine["freq"] == "M"]
        month_ends = pd.PeriodIndex(monthly["period"], freq="M").to_timestamp(how="end").normalize()
        monthly = pd.Series(monthly["value"].to_numpy(), index=month_ends)
        if not daily.empty:
            monthly = monthly[monthly.index < daily.index.min()]
        combined = pd.concat([monthly, daily]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        filled = combined.reindex(combined.index.union(calendar)).ffill().reindex(calendar)
        out[currency] = filled.bfill(limit_area="outside")
    return pd.DataFrame(out)
