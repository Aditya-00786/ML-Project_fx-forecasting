"""Checks that the processed daily data is complete and agrees with an independent source."""

from __future__ import annotations

import numpy as np
import pandas as pd

# FX trades every weekday except these two days, when liquidity is too thin to count.
MARKET_HOLIDAYS = ((12, 25), (1, 1))


def expected_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    holiday = np.zeros(len(days), dtype=bool)
    for month, day in MARKET_HOLIDAYS:
        holiday |= (days.month == month) & (days.day == day)
    return days[~holiday]


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per symbol: date range, share of expected trading days present and thin sessions."""
    rows = {}
    for symbol, frame in panel.groupby(level="symbol"):
        dates = frame.index.get_level_values("date")
        expected = expected_trading_days(dates.min(), dates.max())
        present = expected.isin(dates)
        rows[symbol] = {
            "first": dates.min().date(),
            "last": dates.max().date(),
            "days": len(dates),
            "expected": len(expected),
            "coverage_pct": 100 * present.mean(),
            "missing": int((~present).sum()),
            "thin_days": int(frame["thin"].sum()),
            "median_minutes": float(frame["n_minutes"].median()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def h10_check(panel: pd.DataFrame, macro: pd.DataFrame, pairs: tuple[str, ...]) -> pd.DataFrame:
    """Compare our 12:00 New York prices with the Federal Reserve's H.10 noon rates.

    Both are noon New York quotes, so their daily log returns should be almost identical. The
    match is also reported separately for winter and summer: a timezone error shows up as a
    much weaker match in one season.
    """
    rows = {}
    for pair in pairs:
        ours = panel.xs(pair, level="symbol")["noon"]
        fed = macro[f"h10_{pair}"]
        both = pd.concat({"ours": ours, "fed": fed}, axis=1, sort=True).dropna()
        returns = np.log(both).diff().dropna()
        summer = new_york_dst(returns.index)
        rows[pair] = {
            "days": len(both),
            "return_corr": returns["ours"].corr(returns["fed"]),
            "return_corr_winter": returns[~summer]["ours"].corr(returns[~summer]["fed"]),
            "return_corr_summer": returns[summer]["ours"].corr(returns[summer]["fed"]),
            "mean_abs_level_gap_pct": 100 * (both["ours"] / both["fed"] - 1).abs().mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def new_york_dst(dates: pd.DatetimeIndex) -> np.ndarray:
    """True where noon on that date falls in New York daylight saving time."""
    noon = (dates + pd.Timedelta(hours=12)).tz_localize("America/New_York")
    return np.array([bool(t.dst()) for t in noon])


def largest_moves(panel: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The ``n`` largest absolute daily returns per symbol, for manual inspection."""
    frame = panel.reset_index()[["date", "symbol", "ret"]].dropna()
    frame["abs_ret_pct"] = 100 * frame["ret"].abs()
    top = frame.sort_values("abs_ret_pct", ascending=False).groupby("symbol").head(n)
    top = top.sort_values(["symbol", "abs_ret_pct"], ascending=[True, False])
    return top.reset_index(drop=True)
