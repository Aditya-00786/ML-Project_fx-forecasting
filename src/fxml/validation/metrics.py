"""Forecast accuracy metrics for the three tasks."""

from __future__ import annotations

import numpy as np
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


def direction_metrics(up: np.ndarray, prob_up: np.ndarray) -> dict[str, float]:
    """Accuracy of up/down calls, using a 0.5 threshold on the predicted probability."""
    up, prob_up = np.asarray(up).astype(int), np.asarray(prob_up, dtype=float)
    called = (prob_up > 0.5).astype(int)
    both_classes = len(np.unique(up)) == 2
    return {
        "accuracy": float(np.mean(called == up)),
        "balanced_accuracy": float(balanced_accuracy_score(up, called)),
        "roc_auc": float(roc_auc_score(up, prob_up)) if both_classes else np.nan,
        "brier": float(brier_score_loss(up, np.clip(prob_up, 0, 1))),
        "share_called_up": float(called.mean()),
    }
