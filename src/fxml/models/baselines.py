"""Baseline forecasts that every learning model must beat.

Each function takes one pair's modelling table (indexed by date, as built by
``fxml.features.build``) and its walk-forward folds, and returns forecasts for every test row.
A forecast in row t is for day t+1 and uses only data up to day t: models are re-estimated at
the start of each fold on returns observed before it, then rolled through the fold with their
parameters fixed while their state updates with each new day.

Volatility forecasts are variances of the next day's return, compared with its realized
variance. Return forecasts are log returns. Direction forecasts are probabilities that the next
day's return is positive.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

from fxml.validation.walkforward import Fold

Baseline = Callable[[pd.DataFrame, Sequence[Fold]], pd.Series]


def forecast_positions(folds: Sequence[Fold]) -> np.ndarray:
    return np.concatenate([fold.test for fold in folds])


# ---------------------------------------------------------------- volatility


def vol_naive(df: pd.DataFrame, folds: Sequence[Fold]) -> pd.Series:
    """Tomorrow's variance equals today's realized variance."""
    return df["rv"].iloc[forecast_positions(folds)]


def vol_ewma(df: pd.DataFrame, folds: Sequence[Fold], decay: float = 0.94) -> pd.Series:
    """RiskMetrics: an exponentially weighted average of squared daily returns."""
    variance = (df["ret"].fillna(0) ** 2).ewm(alpha=1 - decay, adjust=False).mean()
    return variance.iloc[forecast_positions(folds)]


def _garch(df: pd.DataFrame, folds: Sequence[Fold], asymmetric: bool) -> pd.Series:
    returns = 100 * df["ret"]  # percent returns keep the optimiser well scaled
    spec = {"mean": "Constant", "vol": "GARCH", "p": 1, "o": int(asymmetric), "q": 1, "dist": "t"}
    params, forecasts = None, []
    for fold in folds:
        history = returns.iloc[: fold.test[0]].dropna()
        result = arch_model(history, **spec).fit(disp="off", show_warning=False)
        if result.convergence_flag == 0 or params is None:
            params = result.params
        through = returns.iloc[: fold.test[-1] + 1].dropna()
        fixed = arch_model(through, **spec).fix(params)
        start = df.index[fold.test[0]]
        variance = fixed.forecast(horizon=1, start=start, reindex=False).variance["h.1"]
        forecasts.append(variance.reindex(df.index[fold.test]) / 1e4)
    return pd.concat(forecasts)


def vol_garch(df: pd.DataFrame, folds: Sequence[Fold]) -> pd.Series:
    """GARCH(1,1) with Student-t errors, estimated by maximum likelihood."""
    return _garch(df, folds, asymmetric=False)


def vol_gjr_garch(df: pd.DataFrame, folds: Sequence[Fold]) -> pd.Series:
    """GJR-GARCH(1,1,1): like GARCH, but falls can raise volatility more than rises."""
    return _garch(df, folds, asymmetric=True)


HAR_COLUMNS = ["logrv_d", "logrv_w", "logrv_m"]


def vol_har(df: pd.DataFrame, folds: Sequence[Fold]) -> pd.Series:
    """HAR-RV: a linear regression of tomorrow's log realized variance on its daily, weekly
    and monthly averages, converted back to variance with a log-normal bias correction."""
    forecasts = []
    for fold in folds:
        train = df.iloc[fold.train].dropna(subset=[*HAR_COLUMNS, "y_logrv"])
        x = np.column_stack([np.ones(len(train)), train[HAR_COLUMNS].to_numpy()])
        beta, *_ = np.linalg.lstsq(x, train["y_logrv"].to_numpy(), rcond=None)
        residual_var = np.var(train["y_logrv"].to_numpy() - x @ beta)
        test = df.iloc[fold.test]
        x_test = np.column_stack([np.ones(len(test)), test[HAR_COLUMNS].to_numpy()])
        forecasts.append(pd.Series(np.exp(x_test @ beta + residual_var / 2), index=test.index))
    return pd.concat(forecasts)


# ---------------------------------------------------------------- returns


def ret_zero(df: pd.DataFrame, folds: Sequence[Fold]) -> pd.Series:
    """The random walk: tomorrow's expected return is zero."""
    return pd.Series(0.0, index=df.index[forecast_positions(folds)])


def ret_mean(df: pd.DataFrame, folds: Sequence[Fold]) -> pd.Series:
    """The average daily return observed before the fold."""
    return pd.concat(
        pd.Series(df["ret"].iloc[: fold.test[0]].mean(), index=df.index[fold.test])
        for fold in folds
    )


def _best_arima_order(history: np.ndarray, max_lag: int) -> tuple[int, int, int]:
    best, best_aic = (0, 0, 0), np.inf
    for p in range(max_lag + 1):
        for q in range(max_lag + 1):
            aic = ARIMA(history, order=(p, 0, q), trend="c").fit().aic
            if aic < best_aic:
                best, best_aic = (p, 0, q), aic
    return best


def ret_arima(df: pd.DataFrame, folds: Sequence[Fold], max_lag: int = 2) -> pd.Series:
    """ARMA(p, q) on daily returns, with orders chosen by AIC once a year (up to ``max_lag``)."""
    returns = 100 * df["ret"]
    order, order_year, forecasts = None, None, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels convergence chatter
        for fold in folds:
            history = returns.iloc[: fold.test[0]].dropna()
            if order_year != fold.test_start.year:
                order = _best_arima_order(history.to_numpy(), max_lag)
                order_year = fold.test_start.year
            result = ARIMA(history.to_numpy(), order=order, trend="c").fit()
            through = returns.iloc[: fold.test[-1] + 1].dropna()
            rolled = result.apply(through.to_numpy())
            # A forecast for the day after date t is the prediction of the next observation.
            first = through.index.get_loc(df.index[fold.test[0]]) + 1
            predicted = rolled.predict(start=first, end=len(through))
            forecasts.append(pd.Series(predicted / 100, index=df.index[fold.test]))
    return pd.concat(forecasts)


# ---------------------------------------------------------------- direction


def dir_frequency(df: pd.DataFrame, folds: Sequence[Fold]) -> pd.Series:
    """The share of up days observed before the fold, every day."""
    return pd.concat(
        pd.Series(float((df["ret"].iloc[: fold.test[0]].dropna() > 0).mean()),
                  index=df.index[fold.test])
        for fold in folds
    )


def dir_persistence(df: pd.DataFrame, folds: Sequence[Fold]) -> pd.Series:
    """Tomorrow moves the way today moved."""
    today = df["ret"].iloc[forecast_positions(folds)]
    return pd.Series(np.select([today > 0, today < 0], [1.0, 0.0], 0.5), index=today.index)


BASELINES: dict[str, dict[str, Baseline]] = {
    "volatility": {
        "naive": vol_naive,
        "ewma": vol_ewma,
        "garch": vol_garch,
        "gjr_garch": vol_gjr_garch,
        "har": vol_har,
    },
    "returns": {"zero": ret_zero, "mean": ret_mean, "arima": ret_arima},
    "direction": {"frequency": dir_frequency, "persistence": dir_persistence},
}
