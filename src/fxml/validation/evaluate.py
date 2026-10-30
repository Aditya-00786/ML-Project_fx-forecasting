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

    ``predictions`` has columns task, pair, model, date, actual and forecast. Days a model or
    the reference did not forecast are left out of that model's scores.
    """
    rows = []
    for (task, pair), group in predictions.groupby(["task", "pair"]):
        wide = group.pivot(index="date", columns="model", values="forecast")
        actual = group.drop_duplicates("date").set_index("date")["actual"].reindex(wide.index)
        ref = reference[task]
        for model in wide.columns:
            # score each model on the days both it and the reference forecast
            ok = actual.notna() & wide[model].notna() & wide[ref].notna()
            truth, forecast, base = actual[ok], wide.loc[ok, model], wide.loc[ok, ref]
            row = {"task": task, "pair": pair, "model": model, "days": int(ok.sum())}
            row |= METRICS[task](truth.to_numpy(), forecast.to_numpy())
            dm, p = diebold_mariano(loss(task, truth, forecast), loss(task, truth, base))
            row |= {f"dm_vs_{ref}": dm, f"dm_p_vs_{ref}": p}
            if task == "direction":
                row["pt_stat"], row["pt_p"] = pesaran_timmermann(truth > 0.5, forecast > 0.5)
            rows.append(row)
    return pd.DataFrame(rows)
