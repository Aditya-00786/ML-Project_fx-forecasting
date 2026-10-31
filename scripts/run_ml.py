"""Run the classical machine learning models through the walk-forward folds and compare them
with the baselines.

1. Builds each pair's modelling frame (Phase 2 features, walk-forward GARCH variance, currency
   factors and regime probabilities) and saves it to data/processed/model_inputs.parquet.
2. Runs every (pair, model) job in parallel. Each job's forecasts are cached in
   reports/predictions/ml/, so an interrupted run continues where it stopped.
3. Scores the forecasts together with the baselines' (reports/predictions/baselines.parquet,
   from run_baselines.py) and writes reports/tables/ml_<task>.csv, ml_regime_vs_global.csv and
   ml_hyperparameters.csv.

    uv run python scripts/run_ml.py [--workers N] [--only TASK/MODEL ...] [--fresh]
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
from fxml.models.inputs import pair_frame  # noqa: E402
from fxml.models.ml import SPECS, get_spec, run_spec  # noqa: E402
from fxml.validation.evaluate import loss, score, summary_tables  # noqa: E402
from fxml.validation.stats_tests import diebold_mariano  # noqa: E402
from fxml.validation.walkforward import walk_forward  # noqa: E402

REFERENCE = {"volatility": "har", "returns": "zero", "direction": "frequency"}
ACTUAL = {"volatility": "y_rv", "returns": "y_ret", "direction": "y_dir"}
SLOW_FIRST = ["svm_rbf", "random_forest", "lightgbm", "knn", "svm_linear"]

_inputs: pd.DataFrame | None = None


def inputs_path(cfg):
    return cfg.paths.processed / "model_inputs.parquet"


def cache_dir(cfg):
    return cfg.paths.reports / "predictions" / "ml"


def build_inputs(cfg) -> pd.DataFrame:
    table = pd.read_parquet(cfg.paths.processed / "features.parquet")
    state = pd.read_parquet(cfg.paths.processed / "market_state.parquet")
    frames = [pair_frame(table, pair, state, cfg.walk_forward.test_start).assign(pair=pair)
              for pair in cfg.pairs]
    return pd.concat(frames).set_index("pair", append=True)


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
    path = cache_dir(cfg) / f"{task}__{name}__{pair}.parquet"
    out.to_parquet(path.with_suffix(".tmp"))
    log.assign(task=task, model=name, pair=pair).to_parquet(
        path.with_name(path.stem + ".log.parquet"))
    path.with_suffix(".tmp").rename(path)  # the forecasts file appears only when complete
    return job, time.perf_counter() - started


def regime_vs_global(predictions: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano test of each regime-aware variant against its global model.

    A negative statistic means the regime-aware variant has the lower loss.
    """
    rows = []
    for (task, pair), group in predictions.groupby(["task", "pair"]):
        wide = group.pivot(index="date", columns="model", values="forecast")
        actual = group.drop_duplicates("date").set_index("date")["actual"].reindex(wide.index)
        for model in wide.columns:
            for suffix in ("_regime", "_experts"):
                base = model.removesuffix(suffix)
                if base == model or base not in wide:
                    continue
                ok = actual.notna() & wide[model].notna() & wide[base].notna()
                stat, p = diebold_mariano(loss(task, actual[ok], wide.loc[ok, model]),
                                          loss(task, actual[ok], wide.loc[ok, base]))
                rows.append({"task": task, "pair": pair, "model": model, "global_model": base,
                             "dm_vs_global": stat, "dm_p_vs_global": p})
    table = pd.DataFrame(rows)
    better = (table["dm_vs_global"] < 0) & (table["dm_p_vs_global"] < 0.05)
    worse = (table["dm_vs_global"] > 0) & (table["dm_p_vs_global"] < 0.05)
    counts = (table.assign(better=better, worse=worse)
              .groupby(["task", "model", "global_model"])[["better", "worse"]].sum()
              .rename(columns={"better": "pairs_significantly_better",
                               "worse": "pairs_significantly_worse"})
              .reset_index().assign(pair="average"))
    return pd.concat([table, counts], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only these models, as TASK/MODEL or MODEL")
    parser.add_argument("--fresh", action="store_true", help="discard cached forecasts first")
    args = parser.parse_args()
    cfg = load_config()

    cache = cache_dir(cfg)
    if args.fresh and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(cfg)
    inputs.to_parquet(inputs_path(cfg), compression="zstd")

    specs = [s for s in SPECS
             if args.only is None or f"{s.task}/{s.name}" in args.only or s.name in args.only]
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

    files = sorted(p for p in cache.glob("*.parquet") if not p.name.endswith(".log.parquet"))
    ml = pd.concat(pd.read_parquet(p) for p in files)
    baselines_file = cfg.paths.reports / "predictions" / "baselines.parquet"
    if not baselines_file.exists():
        raise SystemExit("run scripts/run_baselines.py first: the baselines' forecasts are needed")
    baselines = pd.read_parquet(baselines_file)
    baselines["model"] = baselines["model"].map(lambda m: f"baseline_{m}")
    reference = {task: f"baseline_{name}" for task, name in REFERENCE.items()}
    predictions = pd.concat([baselines, ml], ignore_index=True)
    predictions.to_parquet(cfg.paths.reports / "predictions" / "ml.parquet", compression="zstd")

    tables = cfg.paths.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    for task, table in summary_tables(score(predictions, reference), reference).items():
        table.to_csv(tables / f"ml_{task}.csv", index=False)
        average = table[table["pair"] == "average"].set_index("model").drop(columns="pair")
        print(f"\n{task}: average over pairs")
        print(average.dropna(axis=1, how="all").round(4).to_string())
    comparison = regime_vs_global(ml)
    comparison.to_csv(tables / "ml_regime_vs_global.csv", index=False)
    print("\nregime-aware against global models: pairs significantly better / worse")
    print(comparison[comparison["pair"] == "average"].drop(columns="pair").to_string(index=False))
    logs = pd.concat(pd.read_parquet(p) for p in sorted(cache.glob("*.log.parquet")))
    first = ["task", "model", "pair", "year"]
    logs[first + [c for c in logs.columns if c not in first]].to_csv(
        tables / "ml_hyperparameters.csv", index=False)


if __name__ == "__main__":
    main()

