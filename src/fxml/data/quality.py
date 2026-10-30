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


def coverage_report(panel: pd.DataFrame, start, end) -> pd.DataFrame:
    """Per symbol: how far its data falls short of the configured ``start`` and ``end``, the share
    of expected trading days present in between, and how many sessions are thin or partial."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows = {}
    for symbol, frame in panel.groupby(level="symbol"):
        dates = frame.index.get_level_values("date")
        expected = expected_trading_days(dates.min(), dates.max())
        present = expected.isin(dates)
        rows[symbol] = {
            "first": dates.min().date(),
            "last": dates.max().date(),
            "start_gap_days": len(expected_trading_days(start, dates.min())) - 1,
            "end_gap_days": len(expected_trading_days(dates.max(), end)) - 1,
            "coverage_pct": 100 * present.mean(),
            "missing": int((~present).sum()),
            "thin_days": int(frame["thin"].sum()),
            "partial_days": int(frame["partial"].sum()),
            "median_minutes": float(frame["n_minutes"].median()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def h10_check(panel: pd.DataFrame, macro: pd.DataFrame, pairs: tuple[str, ...]) -> pd.DataFrame:
    """Compare our 12:00 New York prices with the Federal Reserve's H.10 noon rates.

    Both are noon New York quotes, so their daily log returns should be almost identical. The
    match is reported for winter, summer and the weeks when US and EU daylight saving disagree:
    a clock error shows up as a much weaker match in one of these.
    """
    rows = {}
    for pair in pairs:
        ours = panel.xs(pair, level="symbol")["noon"]
        fed = macro[f"h10_{pair}"]
        both = pd.concat({"ours": ours, "fed": fed}, axis=1, sort=True).dropna()
        returns = np.log(both).diff().dropna()
        us_summer = dst_flags(returns.index, "America/New_York")
        clock_gap = us_summer != dst_flags(returns.index, "Europe/Athens")
        rows[pair] = {
            "days": len(both),
            "return_corr": returns["ours"].corr(returns["fed"]),
            "return_corr_winter": _corr(returns[~us_summer]),
            "return_corr_summer": _corr(returns[us_summer]),
            "return_corr_clock_gap": _corr(returns[clock_gap]),
            "clock_gap_days": int(clock_gap.sum()),
            "mean_abs_level_gap_pct": 100 * (both["ours"] / both["fed"] - 1).abs().mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _corr(returns: pd.DataFrame) -> float:
    return returns["ours"].corr(returns["fed"])


def dst_flags(dates: pd.DatetimeIndex, tz: str) -> np.ndarray:
    """True where noon (New York) on that date falls in daylight saving time in ``tz``."""
    noon = (dates + pd.Timedelta(hours=12)).tz_localize("America/New_York").tz_convert(tz)
    return np.array([bool(t.dst()) for t in noon])


def largest_moves(panel: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The ``n`` largest absolute daily returns per symbol, for manual inspection."""
    frame = panel.reset_index()[["date", "symbol", "ret"]].dropna()
    frame["abs_ret_pct"] = 100 * frame["ret"].abs()
    top = frame.sort_values("abs_ret_pct", ascending=False).groupby("symbol").head(n)
    top = top.sort_values(["symbol", "abs_ret_pct"], ascending=[True, False])
    return top.reset_index(drop=True)
