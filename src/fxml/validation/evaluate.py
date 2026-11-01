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


def aligned(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One task and pair's forecasts as a (date x model) table, and the actual values on
    those dates."""
    wide = group.pivot(index="date", columns="model", values="forecast")
    actual = group.drop_duplicates("date").set_index("date")["actual"].reindex(wide.index)
    return wide, actual


def score(
    predictions: pd.DataFrame, reference: dict[str, str], horizon: int = 1
) -> pd.DataFrame:
    """One row per (task, pair, model): accuracy metrics plus a Diebold-Mariano test against
    the task's reference model (negative statistic = better than the reference) and, for
    one-day direction forecasts, a Pesaran-Timmermann test of the up/down calls.

    For forecasts over ``horizon`` days, consecutive forecasts overlap, so the Diebold-Mariano
    test allows for ``horizon - 1`` lags of autocorrelation; Pesaran-Timmermann, which assumes
    independent calls, is skipped.

    ``predictions`` has columns task, pair, model, date, actual and forecast. Days a model or
    the reference did not forecast are left out of that model's scores.
    """
    rows = []
    for (task, pair), group in predictions.groupby(["task", "pair"]):
        wide, actual = aligned(group)
        ref = reference[task]
        for model in wide.columns:
            # score each model on the days both it and the reference forecast
            ok = actual.notna() & wide[model].notna() & wide[ref].notna()
            truth, forecast, base = actual[ok], wide.loc[ok, model], wide.loc[ok, ref]
            row = {"task": task, "pair": pair, "model": model, "days": int(ok.sum())}
            row |= METRICS[task](truth.to_numpy(), forecast.to_numpy())
            dm, p = diebold_mariano(loss(task, truth, forecast), loss(task, truth, base),
                                    horizon)
            row |= {f"dm_vs_{ref}": dm, f"dm_p_vs_{ref}": p}
            if task == "direction" and horizon == 1:
                row["pt_stat"], row["pt_p"] = pesaran_timmermann(truth > 0.5, forecast > 0.5)
            rows.append(row)
    return pd.DataFrame(rows)


def summary_tables(
    scores: pd.DataFrame, reference: dict[str, str]
) -> dict[str, pd.DataFrame]:
    """Per task: one row per (pair, model), plus an "average" row per model.

    Averaging p-values across pairs is meaningless, so the average row instead counts the pairs
    where the model is significantly better than the reference (p < 0.05), or for one-day
    direction forecasts the pairs where its calls beat chance (Pesaran-Timmermann).
    """
    out = {}
    for task, group in scores.groupby("task"):
        group = group.drop(columns="task").dropna(axis=1, how="all")
        dm, p = f"dm_vs_{reference[task]}", f"dm_p_vs_{reference[task]}"
        tests = [dm, p, "pt_stat", "pt_p"]
        metrics = [c for c in group.select_dtypes("number").columns if c not in tests]
        average = group.groupby("model")[metrics].mean()
        if task == "direction" and "pt_p" in group:
            better = group["pt_p"] < 0.05
        else:
            better = (group[dm] < 0) & (group[p] < 0.05)
        average["pairs_significantly_better"] = better.groupby(group["model"]).sum()
        average = average.reset_index().assign(pair="average")
        out[task] = pd.concat([group, average], ignore_index=True)
    return out
