"""Explain the LightGBM volatility and direction models with SHAP values (Phase 6).

Needs data/processed/model_inputs.parquet and reports/tables/ml_hyperparameters.csv from
run_ml.py. Writes reports/tables/shap_lightgbm.csv: the mean absolute and mean SHAP value of
every feature, per task, pair and test year.

    uv run python scripts/run_shap.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import pandas as pd  # noqa: E402

from fxml.config import load_config  # noqa: E402
from fxml.interpret import yearly_shap  # noqa: E402
from fxml.models.ml import get_spec  # noqa: E402
from fxml.validation.walkforward import walk_forward  # noqa: E402

TASKS = ("volatility", "direction")


def run_pair(pair: str) -> pd.DataFrame:
    cfg = load_config()
    frame = pd.read_parquet(cfg.paths.processed / "model_inputs.parquet").xs(pair, level="pair")
    hyper = pd.read_csv(cfg.paths.reports / "tables" / "ml_hyperparameters.csv")
    wf = cfg.walk_forward
    folds = list(walk_forward(frame.index, frame["target_date"], wf.test_start,
                              wf.refit_frequency, wf.validation_fraction, wf.embargo_days))
    frames = []
    for task in TASKS:
        chosen = hyper[(hyper["task"] == task) & (hyper["model"] == "lightgbm")
                       & (hyper["pair"] == pair)]
        params = dict(zip(chosen["year"], chosen["params"], strict=True))
        shap = yearly_shap(get_spec(task, "lightgbm"), frame, folds, params, cfg.seed)
        frames.append(shap.assign(task=task, pair=pair))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cfg = load_config()
    with ProcessPoolExecutor(max_workers=len(cfg.pairs)) as pool:
        shap = pd.concat(pool.map(run_pair, cfg.pairs), ignore_index=True)
    first = ["task", "pair", "year", "group", "feature"]
    shap = shap[first + [c for c in shap.columns if c not in first]]
    shap.to_csv(cfg.paths.reports / "tables" / "shap_lightgbm.csv", index=False)
    pd.set_option("display.width", 200)
    for task in TASKS:
        top = (shap[shap["task"] == task].groupby("feature")["mean_abs_shap"].mean()
               .sort_values(ascending=False).head(10))
        print(f"\n{task}: top features by mean |SHAP|")
        print(top.round(4).to_string())


if __name__ == "__main__":
    main()
