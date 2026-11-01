"""Significance tests for comparing forecasts."""

from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1
) -> tuple[float, float]:
    """Test whether two forecasts have equal expected loss.

    Uses the Harvey-Leybourne-Newbold small-sample correction and a Student-t reference.
    A negative statistic means forecast A has the lower loss. Returns (statistic, two-sided
    p-value).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    centred = d - d.mean()
    long_run_var = centred @ centred / n
    for lag in range(1, horizon):
        long_run_var += 2 * (centred[lag:] @ centred[:-lag]) / n
    if long_run_var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(long_run_var / n)
    dm *= np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def pesaran_timmermann(actual_up: np.ndarray, predicted_up: np.ndarray) -> tuple[float, float]:
    """Test whether direction calls beat what independent guessing would achieve.

    Returns (statistic, one-sided p-value). Undefined (NaN) if the calls never change.
    """
    actual, predicted = np.asarray(actual_up, bool), np.asarray(predicted_up, bool)
    n = len(actual)
    hit_rate = np.mean(actual == predicted)
    py, px = actual.mean(), predicted.mean()
    chance = py * px + (1 - py) * (1 - px)
    var_hit = chance * (1 - chance) / n
    var_chance = (
        (2 * py - 1) ** 2 * px * (1 - px) / n
        + (2 * px - 1) ** 2 * py * (1 - py) / n
        + 4 * py * px * (1 - py) * (1 - px) / n**2
    )
    denominator = var_hit - var_chance
    if denominator <= 0:
        return np.nan, np.nan
    statistic = (hit_rate - chance) / np.sqrt(denominator)
    return float(statistic), float(stats.norm.sf(statistic))


def kupiec(exceeded: np.ndarray, coverage: float) -> tuple[float, float]:
    """Kupiec's proportion-of-failures test for a Value-at-Risk model.

    ``exceeded`` marks days the loss beat the VaR; ``coverage`` is the expected exceedance rate
    (0.01 for a 99% VaR). Returns (likelihood ratio, p-value).
    """
    exceeded = np.asarray(exceeded, bool)
    n, x = len(exceeded), int(exceeded.sum())
    observed = x / n

    def log_lik(p: float) -> float:
        terms = 0.0
        if x < n:
            terms += (n - x) * np.log(1 - p)
        if x > 0:
            terms += x * np.log(p)
        return terms

    ratio = -2 * (log_lik(coverage) - log_lik(observed))
    return float(ratio), float(stats.chi2.sf(ratio, df=1))


def christoffersen(exceeded: np.ndarray) -> tuple[float, float]:
    """Christoffersen's test that Value-at-Risk exceedances are independent from one day to the
    next (they should not cluster). Returns (likelihood ratio, p-value); NaN when there are no
    exceedances or no quiet days to compare."""
    hits = np.asarray(exceeded, bool)
    before, after = hits[:-1], hits[1:]
    n00 = np.sum(~before & ~after)
    n01 = np.sum(~before & after)
    n10 = np.sum(before & ~after)
    n11 = np.sum(before & after)
    if n01 + n11 == 0 or n00 + n10 == 0:
        return np.nan, np.nan

    def log_lik(stay: float, count_stay: int, count_move: int) -> float:
        terms = 0.0
        if count_stay:
            terms += count_stay * np.log(stay)
        if count_move:
            terms += count_move * np.log(1 - stay)
        return terms

    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    restricted = log_lik(1 - p, n00 + n10, n01 + n11)
    unrestricted = log_lik(1 - p01, n00, n01) + log_lik(1 - p11, n10, n11)
    ratio = -2 * (restricted - unrestricted)
    return float(ratio), float(stats.chi2.sf(ratio, df=1))
