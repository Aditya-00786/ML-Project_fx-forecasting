"""Forecasts over longer horizons: the next 5 and 22 trading days (about a week and a month).

For a horizon h, the targets on day t become:

- ``y_rv``: the average daily realized variance over days t+1 to t+h, and ``y_logrv`` its log;
- ``y_ret``: the summed return over those days, and ``y_dir`` whether it is positive;
- ``target_date``: day t+h, when the target is fully known.

The features stay the same, so the Phase 2 baselines and Phase 4 models run unchanged on the
new targets. The walk-forward purges every training row whose target is not fully known before
the test month starts, and the embargo between the fitting rows and the validation slice grows
to h days so their targets never overlap. Forecasts made on consecutive days share most of their
target days, so the significance tests allow for that overlap.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fxml.features.build import TINY
from fxml.models import baselines
from fxml.models.ml import get_spec, run_spec
from fxml.validation.walkforward import Fold, walk_forward

HORIZONS = (5, 22)
VOLATILITY_MODELS = ("naive", "ewma", "garch", "har", "ridge", "lightgbm")
DIRECTION_MODELS = ("frequency", "logistic_l2", "lightgbm")
REFERENCE = {"volatility": "har", "direction": "frequency"}


def horizon_frame(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The modelling frame with its targets replaced by ``horizon``-day ones."""
    out = frame.copy()
    out["y_rv"] = frame["rv"].rolling(horizon).mean().shift(-horizon)
    out["y_logrv"] = np.log(out["y_rv"].clip(lower=TINY))
    out["y_ret"] = frame["ret"].rolling(horizon).sum().shift(-horizon)
    out["y_dir"] = (out["y_ret"] > 0).astype(float).where(out["y_ret"].notna())
    out["target_date"] = pd.Series(frame.index, index=frame.index).shift(-horizon)
    return out


def horizon_folds(frame: pd.DataFrame, horizon: int, wf) -> list[Fold]:
    """Walk-forward folds for ``horizon``-day targets, with an embargo of at least h days."""
    return list(walk_forward(frame.index, frame["target_date"], wf.test_start,
                             wf.refit_frequency, wf.validation_fraction,
                             max(wf.embargo_days, horizon)))


def _test_rows(frame: pd.DataFrame, folds: Sequence[Fold]) -> pd.Index:
    return frame.index[np.concatenate([fold.test for fold in folds])]


def volatility_forecasts(
    frame: pd.DataFrame, folds: Sequence[Fold], horizon: int, seed: int = 42
) -> dict[str, pd.Series]:
    """Average daily variance over the next ``horizon`` days, from every volatility model."""
    test = _test_rows(frame, folds)
    return {
        "naive": frame["rv"].rolling(horizon).mean().reindex(test),  # the last h days' average
        "ewma": baselines.vol_ewma(frame, folds),  # its forecast is flat over any horizon
        "garch": baselines.vol_garch(frame, folds, horizon),
        "har": baselines.vol_har(frame, folds),  # regresses the h-day target directly
        "ridge": run_spec(get_spec("volatility", "ridge"), frame, folds, seed)[0],
        "lightgbm": run_spec(get_spec("volatility", "lightgbm"), frame, folds, seed)[0],
    }


def direction_forecasts(
    frame: pd.DataFrame, folds: Sequence[Fold], seed: int = 42
) -> dict[str, pd.Series]:
    """Probability that the summed return over the horizon is positive."""
    frequency = pd.concat(
        pd.Series(frame["y_dir"].iloc[fold.train].mean(), index=frame.index[fold.test])
        for fold in folds
    )
    return {
        "frequency": frequency,
        "logistic_l2": run_spec(get_spec("direction", "logistic_l2"), frame, folds, seed)[0],
        "lightgbm": run_spec(get_spec("direction", "lightgbm"), frame, folds, seed)[0],
    }


def run_horizon(frame: pd.DataFrame, horizon: int, wf, seed: int = 42) -> pd.DataFrame:
    """Every model's forecasts for one pair and horizon, in the comparison's long format."""
    data = horizon_frame(frame, horizon)
    folds = horizon_folds(data, horizon, wf)
    rows = []
    for task, forecasts, actual in [
        ("volatility", volatility_forecasts(data, folds, horizon, seed), "y_rv"),
        ("direction", direction_forecasts(data, folds, seed), "y_dir"),
    ]:
        for model, forecast in forecasts.items():
            rows.append(pd.DataFrame({
                "task": task, "model": model, "horizon": horizon, "date": forecast.index,
                "actual": data[actual].reindex(forecast.index).to_numpy(),
                "forecast": forecast.to_numpy(),
            }))
    return pd.concat(rows, ignore_index=True)
