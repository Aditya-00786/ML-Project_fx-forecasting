"""Learning curves of Gaussian Naive Bayes against logistic regression (RQ5).

Needs data/processed/model_inputs.parquet from run_ml.py. Writes
reports/predictions/learning_curves.parquet and reports/tables/ml_learning_curves.csv: pooled
metrics per target, model and training size, and the number of pairs where one model's Brier
score is significantly lower than the other's (Diebold-Mariano, p < 0.05).

    uv run python scripts/run_learning_curves.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import pandas as pd  # noqa: E402

from fxml.config import load_config  # noqa: E402
from fxml.models.learning_curves import learning_curve_forecasts, summarise_curves  # noqa: E402
from fxml.validation.stats_tests import diebold_mariano  # noqa: E402
from fxml.validation.walkforward import walk_forward  # noqa: E402


def run_pair(pair: str) -> pd.DataFrame:
    cfg = load_config()
    frame = pd.read_parquet(cfg.paths.processed / "model_inputs.parquet").xs(pair, level="pair")
    wf = cfg.walk_forward
    folds = list(walk_forward(frame.index, frame["target_date"], wf.test_start,
                              wf.refit_frequency, wf.validation_fraction, wf.embargo_days))
    return learning_curve_forecasts(frame, folds).assign(pair=pair)


def significance(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Per target and size: pairs where Naive Bayes or logistic regression has a significantly
    lower Brier score than the other."""
    rows = []
    for (target, size, pair), group in forecasts.groupby(["target", "size", "pair"]):
        wide = group.pivot(index="date", columns="model", values="forecast")
        actual = group.drop_duplicates("date").set_index("date")["actual"].reindex(wide.index)
        stat, p = diebold_mariano((actual - wide["naive_bayes"]) ** 2,
                                  (actual - wide["logistic"]) ** 2)
        rows.append({"target": target, "size": size, "pair": pair,
                     "nb_better": stat < 0 and p < 0.05, "logistic_better": stat > 0 and p < 0.05})
    table = pd.DataFrame(rows).groupby(["target", "size"])[["nb_better", "logistic_better"]].sum()
    return table.rename(columns=lambda c: f"pairs_{c}").reset_index()


def main() -> None:
    cfg = load_config()
    with ProcessPoolExecutor(max_workers=len(cfg.pairs)) as pool:
        forecasts = pd.concat(pool.map(run_pair, cfg.pairs), ignore_index=True)
    out = cfg.paths.reports / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out / "learning_curves.parquet", compression="zstd")
    table = summarise_curves(forecasts).merge(significance(forecasts), on=["target", "size"])
    table = table.sort_values(["target", "size", "model"], key=lambda s: s.replace(0, 10**6))
    table.to_csv(cfg.paths.reports / "tables" / "ml_learning_curves.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
