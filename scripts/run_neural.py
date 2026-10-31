"""Run the neural networks (an MLP and an LSTM per task) through the walk-forward folds.

Each (pair, task, model, seed) job is cached in reports/predictions/neural/, so an interrupted
run continues where it stopped. Networks are refitted at every refit period of the config's
``neural_refit_frequency`` (quarterly by default). Finally every cached model, classical ones
included, is scored with the baselines into reports/tables/ml_*.csv, the neural networks as the
average of their seeds.

Needs data/processed/model_inputs.parquet (built by run_ml.py) and the baselines' forecasts.

    uv run python scripts/run_neural.py [--workers N] [--seeds 3] [--only TASK/MODEL ...] [--fresh]
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
from fxml.models.comparison import (  # noqa: E402
    ACTUAL,
    build_inputs,
    cache_dir,
    inputs_path,
    write_comparison,
    write_job,
)
from fxml.models.neural import NEURAL_SPECS, get_neural_spec, run_neural_spec  # noqa: E402
from fxml.validation.walkforward import walk_forward  # noqa: E402

_inputs: pd.DataFrame | None = None


def job_path(cfg, task: str, name: str, pair: str, seed: int):
    return cache_dir(cfg, "neural") / f"{task}__{name}__{pair}__seed{seed}.parquet"


def run_job(job: tuple[str, str, str, int]) -> tuple[tuple[str, str, str, int], float]:
    """Train one (pair, task, model, seed) through every fold and cache its forecasts."""
    global _inputs
    pair, task, name, seed = job
    cfg = load_config()
    if _inputs is None:
        _inputs = pd.read_parquet(inputs_path(cfg))
    frame = _inputs.xs(pair, level="pair")
    wf = cfg.walk_forward
    folds = list(walk_forward(frame.index, frame["target_date"], wf.test_start,
                              wf.refit_frequency, wf.validation_fraction, wf.embargo_days))
    started = time.perf_counter()
    forecast, log = run_neural_spec(get_neural_spec(task, name), frame, folds,
                                    cfg.seed + seed, wf.neural_refit_frequency)
    out = pd.DataFrame({
        "task": task, "pair": pair, "model": name, "seed": seed, "date": forecast.index,
        "actual": frame[ACTUAL[task]].reindex(forecast.index).to_numpy(),
        "forecast": forecast.to_numpy(),
    })
    write_job(job_path(cfg, task, name, pair, seed), out,
              log.assign(task=task, model=name, pair=pair, seed=seed))
    return job, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only these models, as TASK/MODEL or MODEL")
    parser.add_argument("--fresh", action="store_true", help="discard cached forecasts first")
    args = parser.parse_args()
    cfg = load_config()

    cache = cache_dir(cfg, "neural")
    if args.fresh and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    if not inputs_path(cfg).exists():
        build_inputs(cfg).to_parquet(inputs_path(cfg), compression="zstd")

    specs = [s for s in NEURAL_SPECS
             if args.only is None or f"{s.task}/{s.name}" in args.only or s.name in args.only]
    specs.sort(key=lambda s: s.name != "lstm")  # the slow LSTMs first
    jobs = [(pair, s.task, s.name, seed) for s in specs for pair in cfg.pairs
            for seed in range(args.seeds)
            if not job_path(cfg, s.task, s.name, pair, seed).exists()]
    total = len(specs) * len(cfg.pairs) * args.seeds
    print(f"{len(jobs)} jobs to run ({total - len(jobs)} cached) on {args.workers} workers",
          flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, future in enumerate(as_completed(pool.submit(run_job, j) for j in jobs), 1):
            (pair, task, name, seed), seconds = future.result()
            print(f"[{done}/{len(jobs)} {time.perf_counter() - started:5.0f}s] "
                  f"{task}/{name} {pair} seed {seed}: {seconds:.0f}s", flush=True)
    write_comparison(cfg)


if __name__ == "__main__":
    main()
