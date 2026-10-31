"""Run the classical machine learning models through the walk-forward folds and compare them
with the baselines.

1. Builds each pair's modelling frame (Phase 2 features, walk-forward GARCH variance, currency
   factors and regime probabilities) and saves it to data/processed/model_inputs.parquet.
2. Runs every (pair, model) job in parallel. Each job's forecasts are cached in
   reports/predictions/ml/, so an interrupted run continues where it stopped.
3. Scores every cached model, neural networks included, together with the baselines'
   (reports/predictions/baselines.parquet, from run_baselines.py) and writes
   reports/tables/ml_*.csv (see fxml.models.comparison).

    uv run python scripts/run_ml.py [--workers N] [--only TASK/MODEL ...] [--fresh] [--score-only]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # parallelism comes from running jobs side by side

import argparse  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import pandas as pd  # noqa: E402

from fxml.config import load_config  # noqa: E402
from fxml.models import ml as ml_module  # noqa: E402
from fxml.models.comparison import (  # noqa: E402
    ACTUAL,
    build_inputs,
    cache_dir,
    inputs_path,
    signatures,
    sync_cache,
    write_comparison,
    write_job,
)
from fxml.models.ml import SPECS, get_spec, run_spec  # noqa: E402
from fxml.validation.walkforward import walk_forward  # noqa: E402

SLOW_FIRST = ["svm_rbf", "random_forest", "lightgbm", "knn", "svm_linear"]

_inputs: pd.DataFrame | None = None


def run_job(job: tuple[str, str, str]) -> tuple[tuple[str, str, str], float]:
    """Run one (pair, task, model) job and cache its forecasts and tuning log."""
    global _inputs
    pair, task, name = job
    cfg = load_config()
    if _inputs is None:
        _inputs = pd.read_parquet(inputs_path(cfg))
    frame = _inputs.xs(pair, level="pair")
    wf = cfg.walk_forward
    folds = list(walk_forward(frame.index, frame["target_date"], wf.test_start,
                              wf.refit_frequency, wf.validation_fraction, wf.embargo_days))
    started = time.perf_counter()
    forecast, log = run_spec(get_spec(task, name), frame, folds, cfg.seed)
    out = pd.DataFrame({
        "task": task, "pair": pair, "model": name, "date": forecast.index,
        "actual": frame[ACTUAL[task]].reindex(forecast.index).to_numpy(),
        "forecast": forecast.to_numpy(),
    })
    write_job(cache_dir(cfg, "ml") / f"{task}__{name}__{pair}.parquet", out,
              log.assign(task=task, model=name, pair=pair))
    return job, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only these models, as TASK/MODEL or MODEL")
    parser.add_argument("--fresh", action="store_true", help="discard cached forecasts first")
    parser.add_argument("--score-only", action="store_true",
                        help="only rebuild the comparison tables from cached forecasts")
    args = parser.parse_args()
    cfg = load_config()
    if args.score_only:
        write_comparison(cfg)
        return

    cache = cache_dir(cfg, "ml")
    if args.fresh and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(cfg)
    inputs.to_parquet(inputs_path(cfg), compression="zstd")

    specs = [s for s in SPECS
             if args.only is None or f"{s.task}/{s.name}" in args.only or s.name in args.only]
    for prefix in sync_cache(cache, signatures(specs, inputs, cfg, ml_module)):
        print(f"{prefix}: settings, code or inputs changed, so its cached jobs run again")
    rank = {name: i for i, name in enumerate(SLOW_FIRST)}
    specs.sort(key=lambda s: rank.get(s.name.removesuffix("_regime").removesuffix("_experts"), 99))
    jobs = [(pair, s.task, s.name) for s in specs for pair in cfg.pairs
            if not (cache / f"{s.task}__{s.name}__{pair}.parquet").exists()]
    print(f"{len(jobs)} jobs to run ({len(specs) * len(cfg.pairs) - len(jobs)} cached) "
          f"on {args.workers} workers", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, future in enumerate(as_completed(pool.submit(run_job, j) for j in jobs), 1):
            (pair, task, name), seconds = future.result()
            print(f"[{done}/{len(jobs)} {time.perf_counter() - started:5.0f}s] "
                  f"{task}/{name} {pair}: {seconds:.0f}s", flush=True)
    write_comparison(cfg)


if __name__ == "__main__":
    main()
