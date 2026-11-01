"""Daily 99% Value at Risk from the volatility forecasts, and how often it is exceeded.

Two ways of turning a variance forecast into a Value at Risk:

- ``normal``: the forecast volatility times the normal distribution's 99% quantile;
- ``fhs`` (filtered historical simulation): the forecast volatility times the 1% or 99% quantile
  of past standardised returns (return / forecast volatility) over a trailing window. Fat tails
  and skew in the past errors carry into the Value at Risk.

Both sides are checked: losses on a long position (the return falls below -VaR) and on a short
position (it rises above +VaR). A good 99% Value at Risk is exceeded on 1% of days, with
exceedances that do not cluster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fxml.validation.stats_tests import christoffersen, kupiec

LEVEL = 0.99
FHS_WINDOW = 500  # trading days of past standardised returns
FHS_MIN_DAYS = 250


def value_at_risk(variance: pd.Series, returns: pd.Series, level: float = LEVEL) -> pd.DataFrame:
    """Value at Risk for long and short positions, as positive return thresholds, per day.

    ``variance`` is the forecast of the next day's return variance made at each date and
    ``returns`` the next day's return (both indexed by the forecast date). A return is only
    known one day after its forecast date, so the historical simulation uses standardised
    returns up to the previous date.
    """
    sigma = np.sqrt(variance)
    z = stats.norm.ppf(level)
    standardised = (returns / sigma).shift(1)  # known by the close of each forecast date
    window = standardised.rolling(FHS_WINDOW, min_periods=FHS_MIN_DAYS)
    return pd.DataFrame({
        "normal_long": z * sigma,
        "normal_short": z * sigma,
        "fhs_long": -window.quantile(1 - level) * sigma,
        "fhs_short": window.quantile(level) * sigma,
    })


def exceedances(var: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """True on days the loss beat the Value at Risk (missing where it was not available)."""
    out = {}
    for column in var.columns:
        method, side = column.rsplit("_", 1)
        threshold = var[column]
        hit = returns < -threshold if side == "long" else returns > threshold
        out[column] = hit.where(threshold.notna() & returns.notna())
    return pd.DataFrame(out)


def var_rows(data: pd.DataFrame, pair: str) -> list[dict]:
    """Value at Risk tests for every variance forecast (``var_*`` columns) in one pair's
    backtest data, both methods and sides, compared on the days both methods cover."""
    rows = []
    for column in [c for c in data.columns if c.startswith("var_")]:
        hits = exceedances(value_at_risk(data[column], data["ret"]), data["ret"])
        common = hits.notna().all(axis=1)
        for name in hits.columns:
            method, side = name.rsplit("_", 1)
            rows.append({"model": column.removeprefix("var_"), "pair": pair, "method": method,
                         "side": side} | backtest_var(hits.loc[common, name]))
    return rows


def backtest_var(hits: pd.Series, level: float = LEVEL) -> dict[str, float]:
    """Exceedance rate with the Kupiec (right frequency) and Christoffersen (no clustering)
    tests."""
    hits = hits.dropna().astype(bool).to_numpy()
    lr_pof, p_pof = kupiec(hits, 1 - level)
    lr_ind, p_ind = christoffersen(hits)
    return {"days": len(hits), "exceedances": int(hits.sum()), "rate": float(hits.mean()),
            "kupiec_lr": lr_pof, "kupiec_p": p_pof,
            "christoffersen_lr": lr_ind, "christoffersen_p": p_ind}
