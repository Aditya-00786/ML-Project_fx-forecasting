"""Currency factors: principal components of the seven pairs' daily returns.

Every pair is first expressed as the value of its foreign currency in US dollars, so the returns
of USD/JPY, USD/CHF and USD/CAD change sign. A broad dollar move then pushes all seven the same
way, and the first component reads as a "dollar" factor. Components come from the correlation
matrix, so the more volatile currencies do not dominate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def usd_value_returns(table: pd.DataFrame, pairs: tuple[str, ...]) -> pd.DataFrame:
    """Daily log returns of each foreign currency against the dollar, one column per currency.

    ``table`` is the modelling table indexed by (date, pair) with a ``ret`` column.
    """
    wide = table["ret"].unstack("pair")[list(pairs)]
    out = {}
    for pair in pairs:
        if pair.startswith("USD"):
            out[pair[3:]] = -wide[pair]  # USD/JPY up means the yen fell against the dollar
        else:
            out[pair[:3]] = wide[pair]
    return pd.DataFrame(out)


class CurrencyFactors:
    """Principal components of standardised currency returns, fitted on a training window.

    Signs are fixed so results are comparable across refits: the first component's loadings sum
    to a positive number (all currencies rising against the dollar), and every later component
    has its largest loading positive.
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components

    def fit(self, returns: pd.DataFrame) -> CurrencyFactors:
        clean = returns.dropna()
        self.mean_, self.std_ = clean.mean(), clean.std()
        corr = ((clean - self.mean_) / self.std_).corr().to_numpy()
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        order = np.argsort(eigenvalues)[::-1][: self.n_components]
        components = eigenvectors[:, order].T
        for i, row in enumerate(components):
            flip = row.sum() < 0 if i == 0 else row[np.argmax(np.abs(row))] < 0
            components[i] = -row if flip else row
        names = [f"pc{i + 1}" for i in range(self.n_components)]
        self.loadings_ = pd.DataFrame(components.T, index=returns.columns, columns=names)
        self.explained_variance_ratio_ = pd.Series(
            eigenvalues[order] / eigenvalues.sum(), index=names
        )
        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Factor returns for each day, using only that day's currency returns."""
        standardised = (returns - self.mean_) / self.std_
        return standardised[self.loadings_.index] @ self.loadings_


def average_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Mean pairwise correlation of the columns over a trailing window (missing until every
    pair has a full window)."""
    cols = list(returns.columns)
    pieces = [
        returns[a].rolling(window).corr(returns[b])
        for i, a in enumerate(cols)
        for b in cols[i + 1 :]
    ]
    return pd.concat(pieces, axis=1).mean(axis=1, skipna=False)


def walk_forward_factors(
    returns: pd.DataFrame, first_refit, freq: str = "YS", n_components: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor returns where each day uses loadings fitted only on earlier days.

    Refits at each period start from ``first_refit``; days before it use the first fit's own
    (in-sample) loadings and are flagged in ``in_sample``. Also returns every fit's loadings in
    long form (fit start, currency, one column per component).
    """
    returns = returns.dropna()
    starts = pd.date_range(pd.Timestamp(first_refit), returns.index.max(), freq=freq)
    frames, loadings = [], []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else pd.Timestamp.max
        factors = CurrencyFactors(n_components).fit(returns[returns.index < start])
        period = returns[(returns.index >= start) & (returns.index < stop)]
        frames.append(factors.transform(period).assign(in_sample=False))
        if i == 0:
            before = returns[returns.index < start]
            frames.insert(0, factors.transform(before).assign(in_sample=True))
        loadings.append(factors.loadings_.rename_axis("currency").reset_index().assign(fit=start))
    return pd.concat(frames), pd.concat(loadings, ignore_index=True)
