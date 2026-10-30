"""Build the modelling table: one row per (date, pair) with features known at that day's close
and the next trading day's targets.

Timing rule: a feature for day t uses only information available at 17:00 New York on day t.
Daily bars end exactly then. VIX settles at 16:15, the broad dollar index is a noon rate, Brent
is assessed during the London day, and policy-rate changes are announced in advance, so
same-day values of all of these are known by the close. Macro series are only ever carried
forward, never back.

Holiday sessions (25 December, 1 January) are removed before anything is computed, so no row
or target is a near-empty holiday, and returns are measured across them.

Features that must be estimated inside each training window (GARCH variance, principal
components, regimes) are added later by the models that use them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxml.features import technical as ta

TINY = 1e-12  # floor before taking logs of variances

FEATURE_GROUPS: dict[str, list[str]] = {
    "returns": [
        "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
        "mom_5", "mom_21", "mom_63", "mom_126", "mom_252",
        "ma_dist_20", "ma_dist_50", "ma_dist_200",
    ],
    "technical": ["rsi_14", "macd_hist", "bb_pct_b", "atr_pct", "stoch_k", "stoch_d", "adx_14"],
    "volatility": [
        "logrv_d", "logrv_w", "logrv_m", "log_rv_up", "log_rv_down", "log_parkinson",
        "log_garman_klass", "log_ret_std_5", "log_ret_std_21", "log_ret_std_63", "vol_of_vol_22",
        "partial",
    ],
    "macro": [
        "log_vix", "vix_chg_5", "dollar_ret_1", "dollar_ret_5", "brent_ret_1", "brent_ret_5",
        "gold_ret_1", "gold_ret_5", "spx_ret_1", "spx_ret_5", "rate_diff", "rate_diff_chg_21",
    ],
    "calendar": ["day_of_week", "month_end"],
}
FEATURES = [name for group in FEATURE_GROUPS.values() for name in group]
TARGETS = ["y_rv", "y_logrv", "y_ret", "y_dir", "target_date"]


def trading_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop thin holiday sessions and recompute returns across the days that remain."""
    out = daily[~daily["thin"]].copy()
    out["ret"] = np.log(out["close"]).diff()
    return out


def log_change(series: pd.Series, periods: int) -> pd.Series:
    return np.log(series).diff(periods)


def on_calendar(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Last known value at or before each date (carried forward, never back)."""
    series = series.dropna()
    return series.reindex(series.index.union(dates)).ffill().reindex(dates)


def pair_features(
    daily: pd.DataFrame,
    pair: str,
    macro: pd.DataFrame,
    rates: pd.DataFrame,
    context_close: dict[str, pd.Series],
) -> pd.DataFrame:
    """Features for one pair, indexed by trading date."""
    d = trading_days(daily)
    close, high, low, ret = d["close"], d["high"], d["low"], d["ret"]
    rv = d["rv"].clip(lower=TINY)
    f = pd.DataFrame(index=d.index)

    for lag in range(1, 6):
        f[f"ret_lag{lag}"] = ret.shift(lag - 1)
    for window in (5, 21, 63, 126, 252):
        f[f"mom_{window}"] = log_change(close, window)
    for window in (20, 50, 200):
        f[f"ma_dist_{window}"] = np.log(close / close.rolling(window).mean())

    f["rsi_14"] = ta.rsi(close)
    f["macd_hist"] = ta.macd_histogram(close)
    f["bb_pct_b"] = ta.bollinger_pct_b(close)
    f["atr_pct"] = ta.atr_pct(high, low, close)
    f["stoch_k"], f["stoch_d"] = ta.stochastic(high, low, close)
    f["adx_14"] = ta.adx(high, low, close)

    f["logrv_d"] = np.log(rv)
    f["logrv_w"] = np.log(rv.rolling(5).mean())
    f["logrv_m"] = np.log(rv.rolling(22).mean())
    f["log_rv_up"] = np.log(d["rv_up"].clip(lower=TINY))
    f["log_rv_down"] = np.log(d["rv_down"].clip(lower=TINY))
    f["log_parkinson"] = np.log(d["parkinson"].clip(lower=TINY))
    f["log_garman_klass"] = np.log(d["garman_klass"].clip(lower=TINY))
    for window in (5, 21, 63):
        f[f"log_ret_std_{window}"] = np.log(ret.rolling(window).std().clip(lower=TINY))
    f["vol_of_vol_22"] = f["logrv_d"].rolling(22).std()
    f["partial"] = d["partial"].astype(float)

    dates = d.index
    vix = on_calendar(macro["vix"], dates)
    f["log_vix"] = np.log(vix)
    f["vix_chg_5"] = log_change(vix, 5)
    for name, series in [("dollar", macro["broad_dollar"]), ("brent", macro["brent"]),
                         *context_close.items()]:
        level = on_calendar(series, dates)
        f[f"{name}_ret_1"] = log_change(level, 1)
        f[f"{name}_ret_5"] = log_change(level, 5)
    base, quote = pair[:3], pair[3:]
    diff = on_calendar(rates[base] - rates[quote], dates)
    f["rate_diff"] = diff
    f["rate_diff_chg_21"] = diff.diff(21)

    f["day_of_week"] = dates.day_of_week.astype(float)
    f["month_end"] = ((dates + pd.offsets.BMonthEnd(0)) == dates).astype(float)
    return f.reindex(columns=FEATURES)  # context features stay empty if a symbol is unavailable


def pair_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """Next trading day's realized variance, return and direction, aligned to today's row."""
    d = trading_days(daily)
    nxt = d.shift(-1)
    return pd.DataFrame(
        {
            "y_rv": nxt["rv"],
            "y_logrv": np.log(nxt["rv"].clip(lower=TINY)),
            "y_ret": nxt["ret"],
            "y_dir": (nxt["ret"] > 0).astype(float).where(nxt["ret"].notna()),
            "target_date": pd.Series(d.index, index=d.index).shift(-1),
        },
        index=d.index,
    )


def build_table(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    rates: pd.DataFrame,
    pairs: tuple[str, ...],
    context: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Features, targets and the raw columns baselines need, indexed by (date, pair).

    ``context`` maps a feature prefix to a context symbol in ``panel``, by default gold and the
    S&P 500.
    """
    context = context or {"gold": "XAUUSD", "spx": "SPXUSD"}
    context_close = {
        name: trading_days(panel.xs(symbol, level="symbol"))["close"]
        for name, symbol in context.items()
        if symbol in panel.index.get_level_values("symbol")
    }
    frames = []
    for pair in pairs:
        daily = panel.xs(pair, level="symbol")
        raw = trading_days(daily)[["ret", "rv"]]
        table = pd.concat(
            [pair_features(daily, pair, macro, rates, context_close), pair_targets(daily), raw],
            axis=1,
        )
        frames.append(table.assign(pair=pair))
    return pd.concat(frames).set_index("pair", append=True).sort_index()
