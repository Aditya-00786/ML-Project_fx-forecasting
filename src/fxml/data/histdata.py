"""Download and parse free 1-minute bars from HistData.com.

Files are fetched through the site's public download form and parsed in memory, so raw
ZIP files never touch the disk.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests

BASE_URL = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes"
GET_URL = "https://www.histdata.com/get.php"
USER_AGENT = "Mozilla/5.0 (fxml academic research project)"
FORM_FIELDS = ("tk", "date", "datemonth", "platform", "timeframe", "fxpair")

# HistData's FAQ says its timestamps are EST with no daylight saving, but the data follows New
# York local time: matched against the Fed's H.10 noon rates, our noon returns correlate 0.999 in
# winter and 0.996 in summer when read as New York time, but only 0.944 in summer when read as
# fixed EST. ``scripts/check_data.py`` reports this match by season so a regression shows up.
HISTDATA_TZ = "America/New_York"


@dataclass(frozen=True, order=True)
class Period:
    """One downloadable file: a full year, or a single month of the current year."""

    year: int
    month: int = 0  # 0 means the full-year file

    @property
    def key(self) -> str:
        return f"{self.year}" if self.month == 0 else f"{self.year}-{self.month:02d}"

    def page_url(self, symbol: str) -> str:
        suffix = f"{self.year}" if self.month == 0 else f"{self.year}/{self.month}"
        return f"{BASE_URL}/{symbol.lower()}/{suffix}"


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def available_periods(symbol: str, session: requests.Session) -> list[Period]:
    """List every year and month file HistData offers for ``symbol``."""
    response = session.get(f"{BASE_URL}/{symbol.lower()}", timeout=60)
    response.raise_for_status()
    return parse_period_links(response.text, symbol)


def parse_period_links(html: str, symbol: str) -> list[Period]:
    pattern = rf'1-minute-bar-quotes/{symbol.lower()}/(\d{{4}})(?:/(\d{{1,2}}))?"'
    found = {Period(int(year), int(month or 0)) for year, month in re.findall(pattern, html)}
    return sorted(found)


def select_periods(offered: list[Period], start: date, end: date) -> list[Period]:
    """Files covering ``start`` to ``end``; month files only for years without a full-year file."""
    full_years = {p.year for p in offered if p.month == 0}
    keep = []
    for p in offered:
        if not start.year <= p.year <= end.year:
            continue
        if p.month and (p.year in full_years or (p.year == end.year and p.month > end.month)):
            continue
        keep.append(p)
    return keep


def download_zip(
    symbol: str,
    period: Period,
    session: requests.Session,
    retries: int = 3,
    backoff: float = 5.0,
) -> bytes:
    """Fetch the ZIP for one period, retrying with a growing pause on failure."""
    page_url = period.page_url(symbol)
    for attempt in range(1, retries + 1):
        try:
            page = session.get(page_url, timeout=60)
            page.raise_for_status()
            form = parse_form_fields(page.text)
            response = session.post(GET_URL, data=form, headers={"Referer": page_url}, timeout=300)
            response.raise_for_status()
            if not response.content.startswith(b"PK"):
                raise ValueError(f"{symbol} {period.key}: response is not a ZIP file")
            return response.content
        except (requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
    raise AssertionError("unreachable")


def parse_form_fields(html: str) -> dict[str, str]:
    fields = {}
    for name in FORM_FIELDS:
        match = re.search(rf'name="{name}" id="{name}" value="([^"]*)"', html)
        if match is None:
            raise ValueError(f"download form field {name!r} not found")
        fields[name] = match.group(1)
    return fields


def read_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        return parse_m1_csv(archive.read(name))


def parse_m1_csv(data: bytes | str) -> pd.DataFrame:
    """Parse generic ASCII M1 rows (``YYYYMMDD HHMMSS;open;high;low;close;volume``).

    Returns OHLC columns indexed by the UTC start time of each 1-minute bar. Clock times that do
    not exist or repeat at daylight-saving changes fall on Sunday mornings, when the market is
    closed; any such rows are dropped.
    """
    text = data.decode() if isinstance(data, bytes) else data
    raw = pd.read_csv(
        io.StringIO(text),
        sep=";",
        header=None,
        names=["ts", "open", "high", "low", "close", "volume"],
        dtype={"ts": str},
    )
    local = pd.DatetimeIndex(pd.to_datetime(raw["ts"], format="%Y%m%d %H%M%S"))
    bars = raw[["open", "high", "low", "close"]].astype("float64")
    bars.index = local_to_utc(local)
    return bars[bars.index.notna()].sort_index()


def local_to_utc(local: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert naive HistData clock times to UTC."""
    aware = local.tz_localize(HISTDATA_TZ, ambiguous="NaT", nonexistent="NaT")
    return aware.tz_convert("UTC").rename("ts")


def clean_m1(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate timestamps and rows with impossible prices.

    Returns the cleaned bars and the number of rows removed. Large but genuine moves are
    kept; the quality report lists the biggest ones for manual inspection.
    """
    before = len(bars)
    bars = bars[~bars.index.duplicated(keep="last")]
    body_high = bars[["open", "close"]].max(axis=1)
    body_low = bars[["open", "close"]].min(axis=1)
    valid = (bars > 0).all(axis=1) & (bars["high"] >= body_high) & (bars["low"] <= body_low)
    bars = bars[valid]
    return bars, before - len(bars)
