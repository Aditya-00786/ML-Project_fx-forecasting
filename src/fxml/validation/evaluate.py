"""Score stored forecasts and test them against a reference model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxml.validation.metrics import (
    direction_metrics,
    qlike_losses,
    return_metrics,
    volatility_metrics,
)
from fxml.validation.stats_tests import diebold_mariano, pesaran_timmermann

METRICS = {
    "volatility": volatility_metrics,
    "returns": return_metrics,
    "direction": direction_metrics,
}


def loss(task: str, actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Per-day loss used in Diebold-Mariano tests: QLIKE for volatility, squared error for
    returns, squared probability error (Brier) for direction."""
    if task == "volatility":
        return qlike_losses(actual, forecast)
    return (np.asarray(actual) - np.asarray(forecast)) ** 2


def score(predictions: pd.DataFrame, reference: dict[str, str]) -> pd.DataFrame:
    """One row per (task, pair, model): accuracy metrics plus a Diebold-Mariano test against
    the task's reference model (negative statistic = better than the reference) and, for
    direction, a Pesaran-Timmermann test of the up/down calls.

    ``predictions`` has columns task, pair, model, date, actual and forecast.
    """
    rows = []
    for (task, pair), group in predictions.groupby(["task", "pair"]):
        wide = group.pivot(index="date", columns="model", values="forecast")
        actual = group.drop_duplicates("date").set_index("date")["actual"].reindex(wide.index)
        ref = reference[task]
        for model in wide.columns:
            row = {"task": task, "pair": pair, "model": model, "days": len(actual)}
            row |= METRICS[task](actual.to_numpy(), wide[model].to_numpy())
            dm, p = diebold_mariano(loss(task, actual, wide[model]), loss(task, actual, wide[ref]))
            row |= {f"dm_vs_{ref}": dm, f"dm_p_vs_{ref}": p}
            if task == "direction":
                row["pt_stat"], row["pt_p"] = pesaran_timmermann(actual > 0.5, wide[model] > 0.5)
            rows.append(row)
    return pd.DataFrame(rows)
