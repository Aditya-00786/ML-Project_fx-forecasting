"""Build intraday and daily bars aligned to the FX trading day.

The FX trading day runs from 17:00 to 17:00 New York time, so a bar that starts at 17:00
belongs to the next calendar day and Sunday evening trading belongs to Monday.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resample_bars(m1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute OHLC bars into ``minutes``-minute bars labelled by their start time.

    ``n`` counts the 1-minute bars inside each bar; empty bins are dropped.
    """
    resampler = m1.resample(f"{minutes}min", label="left", closed="left")
    bars = resampler.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    bars["n"] = resampler["close"].count()
    return bars[bars["n"] > 0]


def clip_bad_wicks(bars: pd.DataFrame, max_wick: float = 0.10) -> tuple[pd.DataFrame, int]:
    """Remove highs and lows that reach impossibly far beyond the surrounding prices.

    A wick is measured against the bar's own open and close and its neighbours' prices (previous
    close, next open). Across 2006-2026 every genuine wick, including the 2015 franc shock and
    the 2016 Brexit night, stays under 3%, while bad ticks reach far further (one NZD/USD bar
    shows a high of 2.17 against a price of 0.57). Wicks beyond ``max_wick`` are cut back to the
    bar's body. Returns the cleaned bars and the number of bars changed.
    """
    neighbours = pd.concat(
        [bars["open"], bars["close"], bars["close"].shift(1), bars["open"].shift(-1)], axis=1
    )
    ceiling = neighbours.max(axis=1) * (1 + max_wick)
    floor = neighbours.min(axis=1) * (1 - max_wick)
    bad_high = bars["high"] > ceiling
    bad_low = bars["low"] < floor
    out = bars.copy()
    out.loc[bad_high, "high"] = out.loc[bad_high, ["open", "close"]].max(axis=1)
    out.loc[bad_low, "low"] = out.loc[bad_low, ["open", "close"]].min(axis=1)
    return out, int((bad_high | bad_low).sum())


def trading_dates(index: pd.DatetimeIndex, tz: str, close_hour: int) -> pd.DatetimeIndex:
    """Map UTC bar start times to the trading date they belong to.

    Uses local wall-clock time so the boundary stays at ``close_hour`` through daylight saving.
    """
    local = index.tz_convert(tz).tz_localize(None)
    return (local + pd.Timedelta(hours=24 - close_hour)).normalize()


def daily_bars(bars: pd.DataFrame, tz: str, close_hour: int) -> pd.DataFrame:
    """Summarise intraday bars into one row per trading day (weekend sessions dropped).

    Columns: open, high, low, close, noon (last price at or before 12:00 local time),
    n_bars, n_minutes (1-minute bars covered) and rv (realized variance: the sum of squared
    intraday log returns, where the first bar of each day uses its own open).
    """
    local = bars.index.tz_convert(tz).tz_localize(None)
    day = (local + pd.Timedelta(hours=24 - close_hour)).normalize()
    frame = bars.assign(day=day, local=local)
    frame = frame[frame["day"].dt.day_of_week < 5]

    log_close = np.log(frame["close"])
    previous = log_close.groupby(frame["day"]).shift(1)
    first_return = np.log(frame["close"] / frame["open"])
    intraday_return = (log_close - previous).fillna(first_return)
    frame = frame.assign(r2=intraday_return**2)

    grouped = frame.groupby("day")
    out = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "n_bars": grouped["close"].size(),
            "n_minutes": grouped["n"].sum(),
            "rv": grouped["r2"].sum(),
        }
    )
    morning = frame[(frame["local"].dt.hour < 12) & (frame["local"].dt.normalize() == frame["day"])]
    out["noon"] = morning.groupby("day")["close"].last()
    out.index.name = "date"
    return out


def add_daily_measures(daily: pd.DataFrame, minutes_per_day: int = 1440) -> pd.DataFrame:
    """Add close-to-close log returns, range-based variance estimators and a thin-session flag.

    ``thin`` marks sessions covering under half the minutes of a full day, such as the
    25 December and 1 January holidays.
    """
    out = daily.copy()
    log_hl = np.log(out["high"] / out["low"])
    log_co = np.log(out["close"] / out["open"])
    out["ret"] = np.log(out["close"]).diff()
    out["parkinson"] = log_hl**2 / (4 * np.log(2))
    out["garman_klass"] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    out["thin"] = out["n_minutes"] < 0.5 * minutes_per_day
    return out
