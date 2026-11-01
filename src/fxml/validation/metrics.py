"""Forecast accuracy metrics for the three tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


def qlike_losses(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """QLIKE loss per day: zero for a perfect forecast and robust to noise in the realized proxy.

    It penalises under-predicting variance more than over-predicting it.
    """
    ratio = np.asarray(realized) / np.asarray(forecast)
    return ratio - np.log(ratio) - 1


def volatility_metrics(realized: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    """Accuracy of variance forecasts against realized variance.

    ``mz_r2`` and ``mz_slope`` come from the Mincer-Zarnowitz regression of log realized on log
    forecast variance; an unbiased forecast has slope one.
    """
    realized, forecast = np.asarray(realized), np.asarray(forecast)
    log_error = np.log(realized) - np.log(forecast)
    slope, intercept = np.polyfit(np.log(forecast), np.log(realized), 1)
    fitted = intercept + slope * np.log(forecast)
    residual = np.log(realized) - fitted
    total = np.log(realized) - np.log(realized).mean()
    return {
        "qlike": float(qlike_losses(realized, forecast).mean()),
        "rmse_log": float(np.sqrt(np.mean(log_error**2))),
        "mae_log": float(np.mean(np.abs(log_error))),
        "mz_r2": float(1 - residual @ residual / (total @ total)),
        "mz_slope": float(slope),
    }


def return_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    """Accuracy of return forecasts. ``r2_oos`` compares against always forecasting zero."""
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    error = actual - forecast
    called = forecast != 0
    hits = np.sign(forecast[called]) == np.sign(actual[called])
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2_oos": float(1 - error @ error / (actual @ actual)),
        "sign_hit_rate": float(hits.mean()) if called.any() else np.nan,
    }


def calibration_bins(actual: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability table: forecasts split into equal-count bins, with the average forecast and
    the share of days the event happened in each. A calibrated model has the two equal.

    Equal-count bins suit forecasts bunched near 50%, as direction forecasts are. Forecasts that
    take only a few values (at most ``bins``) get one bin per value instead."""
    actual, prob = np.asarray(actual, dtype=float), np.asarray(prob, dtype=float)
    values = np.unique(prob)
    if len(values) <= bins:
        which = np.searchsorted(values, prob)
    else:
        edges = np.unique(np.quantile(prob, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    table = pd.DataFrame({"bin": which, "forecast": prob, "actual": actual})
    return (table.groupby("bin").agg(mean_forecast=("forecast", "mean"),
                                     frequency=("actual", "mean"), days=("actual", "size"))
            .reset_index(drop=True))


def expected_calibration_error(actual: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """Day-weighted average gap between forecast and frequency over the reliability bins."""
    table = calibration_bins(actual, prob, bins)
    gap = (table["mean_forecast"] - table["frequency"]).abs()
    return float((gap * table["days"]).sum() / table["days"].sum())


def calibration_slope(actual: np.ndarray, prob: np.ndarray, steps: int = 50) -> float:
    """Slope of a logistic regression of the outcome on the forecast's log-odds.

    1 means the forecasts are as confident as they should be, below 1 overconfident (the
    outcomes vary less than the forecasts claim) and above 1 underconfident. Missing when the
    forecasts barely vary, since the slope is then not identified."""
    y = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    if np.std(x) < 1e-6:
        return np.nan
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(steps):  # Newton-Raphson for the two coefficients
        fitted = 1 / (1 + np.exp(-design @ beta))
        weight = fitted * (1 - fitted)
        hessian = design.T @ (design * weight[:, None])
        if np.linalg.cond(hessian) > 1e12:
            return np.nan
        step = np.linalg.solve(hessian, design.T @ (y - fitted))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[1])


def direction_metrics(up: np.ndarray, prob_up: np.ndarray) -> dict[str, float]:
    """Accuracy of up/down calls, using a 0.5 threshold on the predicted probability, and how
    well calibrated the probabilities are."""
    up, prob_up = np.asarray(up).astype(int), np.asarray(prob_up, dtype=float)
    called = (prob_up > 0.5).astype(int)
    both_classes = len(np.unique(up)) == 2
    return {
        "accuracy": float(np.mean(called == up)),
        "balanced_accuracy": float(balanced_accuracy_score(up, called)),
        "roc_auc": float(roc_auc_score(up, prob_up)) if both_classes else np.nan,
        "brier": float(brier_score_loss(up, np.clip(prob_up, 0, 1))),
        "share_called_up": float(called.mean()),
        "ece": expected_calibration_error(up, prob_up),
        "calibration_slope": calibration_slope(up, prob_up),
    }
