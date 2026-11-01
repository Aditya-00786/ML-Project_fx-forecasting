"""What drives the LightGBM forecasts: SHAP values from LightGBM's own TreeSHAP.

A SHAP value splits one forecast into a contribution from each feature, so that the
contributions plus the average forecast add up to the forecast itself. Averaging their size
over many days ranks the features.

For each pair and year, the model the walk-forward used that January is refitted (same
training rows, the hyperparameters tuned for that year) and explained on that year's test days.
The monthly refits later in the year are close variants of it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fxml.features.build import FEATURE_GROUPS
from fxml.models import ml
from fxml.models.inputs import FACTOR_COLUMNS
from fxml.validation.walkforward import Fold

GROUP_OF = {feature: group for group, features in FEATURE_GROUPS.items() for feature in features}
GROUP_OF |= {"garch_logvar": "volatility"} | dict.fromkeys(FACTOR_COLUMNS, "currency factors")


def contributions(model, x: np.ndarray) -> np.ndarray:
    """SHAP values, one column per feature (the expected value column is dropped).

    For a classifier they are in log-odds of an up day; for a regressor, in the target's units.
    """
    return np.asarray(model.predict(x, pred_contrib=True))[:, :-1]


def yearly_shap(
    spec: ml.Spec, frame: pd.DataFrame, folds: Sequence[Fold], params_by_year: dict[int, str],
    seed: int = 42,
) -> pd.DataFrame:
    """Mean absolute and mean SHAP value of every feature, per test year.

    ``params_by_year`` maps a year to the tuned hyperparameters (JSON, as logged by run_ml.py).
    Years without tuned hyperparameters are skipped.
    """
    columns = ml.feature_columns(spec, frame)
    rows = []
    for year in sorted({f.test_start.year for f in folds} & set(params_by_year)):
        in_year = [f for f in folds if f.test_start.year == year]
        params = json.loads(params_by_year[year])
        model = ml.fit(spec, params, frame.iloc[in_year[0].train], seed)[None]
        test = frame.iloc[np.concatenate([f.test for f in in_year])]
        shap = contributions(model, test[columns].to_numpy(dtype=float))
        rows.append(pd.DataFrame({
            "year": year, "feature": columns,
            "mean_abs_shap": np.abs(shap).mean(axis=0), "mean_shap": shap.mean(axis=0),
        }))
    if not rows:
        return pd.DataFrame(columns=["year", "feature", "mean_abs_shap", "mean_shap", "group"])
    out = pd.concat(rows, ignore_index=True)
    out["group"] = out["feature"].map(GROUP_OF).fillna("other")
    return out
