"""Standard technical indicators from daily OHLC prices.

Every indicator uses trailing windows or recursive smoothing only, so its value on a day depends
on that day and earlier days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, an exponential average with weight 1/period.

    As Wilder defined it, the average starts from the simple mean of the first ``period``
    values and is missing before that, so the first days of the data do not dominate it.
    """
    valid = np.flatnonzero(series.notna().to_numpy())
    if len(valid) < period:
        return pd.Series(np.nan, index=series.index)
    seed = valid[period - 1]
    values = series.astype(float).copy()
    values.iloc[:seed] = np.nan
    values.iloc[seed] = series.iloc[valid[:period]].mean()
    return values.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = wilder(change.clip(lower=0), period)
    loss = wilder(-change.clip(upper=0), period)
    return 100 - 100 / (1 + gain / loss)


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line minus its signal line, divided by price so pairs are comparable."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return (line - line.ewm(span=signal, adjust=False).mean()) / close


def bollinger_pct_b(close: pd.Series, window: int = 20, width: float = 2.0) -> pd.Series:
    """Where the price sits within its Bollinger bands: 0 at the lower band, 1 at the upper."""
    mean, std = close.rolling(window).mean(), close.rolling(window).std()
    return (close - (mean - width * std)) / (2 * width * std)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous = close.shift(1)
    ranges = [high - low, (high - previous).abs(), (low - previous).abs()]
    return pd.concat(ranges, axis=1).max(axis=1)


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range as a share of price."""
    return wilder(true_range(high, low, close), period) / close


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth: int = 3
) -> tuple[pd.Series, pd.Series]:
    """%K (position of the close in the recent high-low range) and its moving average %D."""
    lowest, highest = low.rolling(period).min(), high.rolling(period).max()
    k = (close - lowest) / (highest - lowest)
    return k, k.rolling(smooth).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average directional index: trend strength from 0 (none) to 100, ignoring direction."""
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)
    tr = wilder(true_range(high, low, close), period)
    plus_di = 100 * wilder(plus_dm, period) / tr
    minus_di = 100 * wilder(minus_dm, period) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return wilder(dx, period)
