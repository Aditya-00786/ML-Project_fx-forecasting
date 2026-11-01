"""Forecast volatility and direction over the next 5 and 22 trading days (an extension).

Needs data/processed/model_inputs.parquet (from run_ml.py). Runs every (pair, horizon) in
parallel and writes reports/predictions/horizons.parquet and
reports/tables/horizons_<task>.csv: metrics per pair, model and horizon, with Diebold-Mariano
tests against HAR-RV (volatility) and the historical frequency of up periods (direction) that
allow for the overlap of consecutive forecasts.

    uv run python scripts/run_horizons.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import pandas as pd  # noqa: E402

from fxml.config import load_config  # noqa: E402
from fxml.models.horizons import HORIZONS, REFERENCE, run_horizon  # noqa: E402
from fxml.validation.evaluate import score, summary_tables  # noqa: E402


def run_job(job: tuple[str, int]) -> pd.DataFrame:
    pair, horizon = job
    cfg = load_config()
    frame = pd.read_parquet(cfg.paths.processed / "model_inputs.parquet").xs(pair, level="pair")
    out = run_horizon(frame, horizon, cfg.walk_forward, cfg.seed).assign(pair=pair)
    print(f"{pair} {horizon}-day: {len(out):,} forecasts", flush=True)
    return out


def main() -> None:
    cfg = load_config()
    jobs = [(pair, h) for h in HORIZONS for pair in cfg.pairs]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        predictions = pd.concat(pool.map(run_job, jobs), ignore_index=True)
    predictions.to_parquet(cfg.paths.reports / "predictions" / "horizons.parquet",
                           compression="zstd")
    parts = {}
    for horizon, group in predictions.groupby("horizon"):
        tables = summary_tables(score(group.drop(columns="horizon"), REFERENCE, horizon),
                                REFERENCE)
        for task, table in tables.items():
            parts.setdefault(task, []).append(table.assign(horizon=horizon))
    pd.set_option("display.width", 200)
    for task, frames in parts.items():
        table = pd.concat(frames, ignore_index=True)
        first = ["horizon", "pair", "model"]
        table = table[first + [c for c in table.columns if c not in first]]
        table.to_csv(cfg.paths.reports / "tables" / f"horizons_{task}.csv", index=False)
        average = table[table["pair"] == "average"].drop(columns="pair")
        print(f"\n{task}: average over pairs")
        print(average.dropna(axis=1, how="all").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
