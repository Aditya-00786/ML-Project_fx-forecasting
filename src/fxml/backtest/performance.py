"""Performance of daily strategy returns: return, risk, drawdowns, costs and significance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DAYS_PER_YEAR = 252
HAC_LAGS = 5  # Newey-West lags for the t-test of the mean daily return


def newey_west_t(returns: np.ndarray, lags: int = HAC_LAGS) -> tuple[float, float]:
    """t statistic of the mean with Newey-West standard errors, and its two-sided p-value."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    centred = x - x.mean()
    variance = centred @ centred / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * (centred[lag:] @ centred[:-lag]) / n
    if n < 2 or variance <= 0:
        return np.nan, np.nan
    t = x.mean() / np.sqrt(variance / n)
    return float(t), float(2 * stats.t.sf(abs(t), df=n - 1))


def max_drawdown(returns: pd.Series) -> float:
    """Largest fall of cumulative return from its previous peak (a negative number)."""
    wealth = returns.fillna(0.0).cumsum()
    return float((wealth - wealth.cummax().clip(lower=0.0)).min())


def summary(daily: pd.DataFrame, column: str = "net") -> dict[str, float]:
    """Annualised performance of one strategy's daily returns."""
    returns = daily[column]
    vol = returns.std() * np.sqrt(DAYS_PER_YEAR)
    t, p = newey_west_t(returns.to_numpy())
    invested = daily["position"] != 0
    return {
        "days": int(len(returns)),
        "ann_return": float(returns.mean() * DAYS_PER_YEAR),
        "ann_vol": float(vol),
        "sharpe": float(returns.mean() * DAYS_PER_YEAR / vol) if vol > 0 else np.nan,
        "max_drawdown": max_drawdown(returns),
        "hit_rate": float((returns[invested] > 0).mean()) if invested.any() else np.nan,
        "share_invested": float(invested.mean()),
        "turnover": float(daily["turnover"].mean() * DAYS_PER_YEAR),
        "cost_drag": float(daily["cost"].mean() * DAYS_PER_YEAR),
        "t_stat": t,
        "p_value": p,
    }


def portfolio(results: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average over pairs, day by day, for each strategy. With volatility
    targeting every pair carries the same forecast risk, so this is an equal-risk portfolio."""
    columns = ["position", "turnover", "gross", "cost", "net",
               *[c for c in results.columns if c.startswith("net_")]]
    grouped = results.groupby(["signal", "sizing", "date"])
    out = grouped[columns].mean()
    out["position"] = results.assign(position=results["position"].abs()).groupby(
        ["signal", "sizing", "date"])["position"].mean()  # gross exposure
    return out.reset_index()


def strategy_table(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """One row per strategy (and ``by`` group): performance at normal and stressed costs."""
    stressed = [c for c in results.columns if c.startswith("net_")]
    rows = []
    for keys, daily in results.groupby(["signal", "sizing", *by]):
        daily = daily.sort_values("date")
        row = dict(zip(["signal", "sizing", *by], keys, strict=True)) | summary(daily)
        row["gross_sharpe"] = summary(daily, "gross")["sharpe"]
        for column in stressed:
            row[f"sharpe_{column.removeprefix('net_')}_costs"] = summary(daily, column)["sharpe"]
        rows.append(row)
    return pd.DataFrame(rows)


def breakdown(results: pd.DataFrame, buckets: pd.Series, name: str) -> pd.DataFrame:
    """Performance of each strategy within each bucket of days (a regime, a stress episode,
    a subperiod). ``buckets`` maps each date to its bucket."""
    labelled = results.assign(**{name: buckets.reindex(results["date"]).to_numpy()})
    labelled = labelled.dropna(subset=[name])
    return strategy_table(labelled, [name])


def risk_targeting(results: pd.DataFrame, target: float) -> pd.DataFrame:
    """How closely each sizing rule hits the volatility target, quarter by quarter.

    Uses gross returns of the always-long signal, so only the sizing differs. ``tracking`` is
    the root mean square of log(realised quarterly volatility / target); 0 is a perfect hit.
    """
    long = results[results["signal"] == "long"].copy()
    long["quarter"] = pd.PeriodIndex(long["date"], freq="Q")
    quarterly = (long.groupby(["pair", "sizing", "quarter"])["gross"].std()
                 * np.sqrt(DAYS_PER_YEAR)).rename("realised_vol").reset_index()
    quarterly["log_ratio"] = np.log(quarterly["realised_vol"] / target)
    return (quarterly.groupby(["pair", "sizing"])
            .agg(mean_realised_vol=("realised_vol", "mean"),
                 tracking=("log_ratio", lambda x: float(np.sqrt(np.mean(x ** 2)))),
                 quarters_above_1_5x=("realised_vol", lambda v: float((v > 1.5 * target).mean())))
            .reset_index())
