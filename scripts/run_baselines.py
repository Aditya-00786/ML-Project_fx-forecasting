"""Run every baseline through the walk-forward folds for all pairs and score the results.

Writes reports/predictions/baselines.parquet (every forecast) and
reports/tables/baselines_<task>.csv (metrics per pair and model, plus the average over pairs).
Pairs run in parallel.

    uv run python scripts/run_baselines.py
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from fxml.config import load_config
from fxml.models.baselines import BASELINES
from fxml.validation.evaluate import score, summary_tables
from fxml.validation.walkforward import walk_forward

TARGET = {"volatility": "y_rv", "returns": "y_ret", "direction": "y_dir"}
REFERENCE = {"volatility": "naive", "returns": "zero", "direction": "frequency"}


def run_pair(pair: str) -> pd.DataFrame:
    cfg = load_config()
    table = pd.read_parquet(cfg.paths.processed / "features.parquet")
    df = table.xs(pair, level="pair")
    wf = cfg.walk_forward
    folds = list(walk_forward(df.index, df["target_date"], wf.test_start, wf.refit_frequency,
                              wf.validation_fraction, wf.embargo_days))
    frames = []
    for task, models in BASELINES.items():
        for name, model in models.items():
            started = time.perf_counter()
            forecast = model(df, folds)
            frames.append(pd.DataFrame({
                "task": task, "pair": pair, "model": name, "date": forecast.index,
                "actual": df[TARGET[task]].reindex(forecast.index).to_numpy(),
                "forecast": forecast.to_numpy(),
            }))
            print(f"{pair} {task}/{name}: {len(forecast):,} forecasts in "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-only", action="store_true",
                        help="re-score saved forecasts without running the models again")
    args = parser.parse_args()
    cfg = load_config()
    out = cfg.paths.reports / "predictions"
    if args.score_only:
        predictions = pd.read_parquet(out / "baselines.parquet")
    else:
        with ProcessPoolExecutor(max_workers=len(cfg.pairs)) as pool:
            predictions = pd.concat(pool.map(run_pair, cfg.pairs), ignore_index=True)
        out.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(out / "baselines.parquet", compression="zstd")

    tables = cfg.paths.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    for task, table in summary_tables(score(predictions, REFERENCE), REFERENCE).items():
        table.to_csv(tables / f"baselines_{task}.csv", index=False)
        average = table[table["pair"] == "average"].set_index("model")
        print(f"\n{task}: average over pairs")
        print(average.drop(columns="pair").dropna(axis=1, how="all").round(4).to_string())


if __name__ == "__main__":
    main()
