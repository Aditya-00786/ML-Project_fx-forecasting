"""Forecast and backtest the extra pairs (a stretch goal) with the core walk-forward protocol.

Needs data/processed/extra_inputs.parquet and extra_daily_bars.parquet from
build_extra_pairs.py. Each pair gets the baselines, a representative set of the classical models
and the MLP (three seeds, averaged), on the same folds, tuning and refits as the core pairs. A
daily-only pair has no realized variance, so it gets no volatility task: only its EWMA and
GARCH variance forecasts are made, for position sizing and Value at Risk.

Writes:

- reports/predictions/extra_pairs.parquet: every forecast;
- reports/tables/extra_pairs_<task>.csv: metrics per pair and model, with Diebold-Mariano tests
  against HAR-RV, the zero return and the frequency of up days (and Pesaran-Timmermann tests of
  the direction calls), as in the core comparison;
- reports/tables/extra_pairs_backtest.csv: every strategy per pair, net of costs;
- reports/tables/extra_pairs_var.csv: 99% Value at Risk exceedances and their tests.

    uv run python scripts/run_extra_pairs.py --workers 8
    uv run python scripts/run_extra_pairs.py --score-only   # re-score and re-run the backtest
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import pandas as pd  # noqa: E402

from fxml.backtest.performance import strategy_table  # noqa: E402
from fxml.backtest.risk import var_rows as risk_rows  # noqa: E402
from fxml.backtest.strategies import pair_data, run_pair  # noqa: E402
from fxml.config import load_config  # noqa: E402
from fxml.features.build import trading_days  # noqa: E402
from fxml.models.baselines import BASELINES  # noqa: E402
from fxml.models.comparison import ACTUAL, REFERENCE, seed_average  # noqa: E402
from fxml.models.ml import get_spec, run_spec  # noqa: E402
from fxml.models.neural import get_neural_spec, run_neural_spec  # noqa: E402
from fxml.validation.evaluate import score, summary_tables  # noqa: E402
from fxml.validation.walkforward import walk_forward  # noqa: E402

MODELS = {  # family -> task -> models
    "baseline": {"volatility": ("naive", "ewma", "garch", "gjr_garch", "har"),
                 "returns": ("zero", "mean", "arima"),
                 "direction": ("frequency", "persistence")},
    "ml": {"volatility": ("ridge", "random_forest", "lightgbm"),
           "returns": ("ridge", "lightgbm"),
           "direction": ("logistic_l2", "naive_bayes", "random_forest", "lightgbm")},
    "neural": {"volatility": ("mlp",), "returns": ("mlp",), "direction": ("mlp",)},
}
DAILY_ONLY_VOLATILITY = ("ewma", "garch")  # need only daily returns; for sizing and VaR
SEEDS = 3

_inputs: pd.DataFrame | None = None


def jobs_for(pair: str, daily_only: bool) -> list[tuple]:
    out = []
    for family, tasks in MODELS.items():
        for task, models in tasks.items():
            if daily_only and task == "volatility":
                models = DAILY_ONLY_VOLATILITY if family == "baseline" else ()
            for model in models:
                for seed in range(SEEDS if family == "neural" else 1):
                    out.append((pair, family, task, model, seed))
    return out


def run_job(job: tuple) -> pd.DataFrame:
    global _inputs
    pair, family, task, model, seed = job
    cfg = load_config()
    if _inputs is None:
        _inputs = pd.read_parquet(cfg.paths.processed / "extra_inputs.parquet")
    frame = _inputs.xs(pair, level="pair")
    wf = cfg.walk_forward
    folds = list(walk_forward(frame.index, frame["target_date"], wf.test_start,
                              wf.refit_frequency, wf.validation_fraction, wf.embargo_days))
    if family == "baseline":
        forecast, name = BASELINES[task][model](frame, folds), f"baseline_{model}"
    elif family == "ml":
        forecast, name = run_spec(get_spec(task, model), frame, folds, cfg.seed)[0], model
    else:
        forecast, name = run_neural_spec(get_neural_spec(task, model), frame, folds,
                                         cfg.seed + seed, wf.neural_refit_frequency)[0], model
    return pd.DataFrame({
        "task": task, "pair": pair, "model": name, "seed": seed, "date": forecast.index,
        "actual": frame[ACTUAL[task]].reindex(forecast.index).to_numpy(),
        "forecast": forecast.to_numpy(),
    })


def forecast_all(cfg, workers: int) -> pd.DataFrame:
    jobs = [job for pair in cfg.extra_pairs for job in jobs_for(pair, pair in cfg.daily_only)]
    jobs.sort(key=lambda j: (j[1] != "neural", j[1] != "ml"))  # the slow networks first
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    started, frames = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            frames.append(future.result())
            pair, family, task, model, seed = futures[future]
            print(f"[{done}/{len(jobs)} {time.perf_counter() - started:5.0f}s] "
                  f"{pair} {task}/{model} seed {seed}", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    neural = raw["model"].isin([m for ms in MODELS["neural"].values() for m in ms])
    return pd.concat([raw[~neural].drop(columns="seed"), seed_average(raw[neural])],
                     ignore_index=True)


def backtest(cfg, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every strategy per pair, and Value at Risk exceedances per variance forecast."""
    inputs = pd.read_parquet(cfg.paths.processed / "extra_inputs.parquet")
    bars = pd.read_parquet(cfg.paths.processed / "extra_daily_bars.parquet")
    results, var = [], []
    for pair in cfg.extra_pairs:
        frame = inputs.xs(pair, level="pair")
        close = trading_days(bars.xs(pair, level="symbol"))["close"]
        data = pair_data(frame, close, predictions[predictions["pair"] == pair],
                         cfg.costs.spread_in_price(pair))
        results.append(run_pair(data, (1.0, cfg.costs.stress_multiplier)).assign(pair=pair))
        var += risk_rows(data, pair)
    return strategy_table(pd.concat(results, ignore_index=True), ["pair"]), pd.DataFrame(var)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--score-only", action="store_true",
                        help="re-score saved forecasts and re-run the backtest")
    args = parser.parse_args()
    cfg = load_config()
    saved = cfg.paths.reports / "predictions" / "extra_pairs.parquet"
    if args.score_only:
        predictions = pd.read_parquet(saved)
    else:
        predictions = forecast_all(cfg, args.workers)
        saved.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(saved, compression="zstd")

    reference = {task: f"baseline_{name}" for task, name in REFERENCE.items()}
    scored = predictions[predictions["actual"].notna().groupby(
        [predictions["task"], predictions["pair"]]).transform("any")]  # not daily-only volatility
    tables = cfg.paths.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    for task, table in summary_tables(score(scored, reference), reference).items():
        table.to_csv(tables / f"extra_pairs_{task}.csv", index=False)
        print(f"\n{task}")
        print(table.set_index(["pair", "model"]).dropna(axis=1, how="all").round(4).to_string())

    strategies, var = backtest(cfg, predictions)
    strategies.to_csv(tables / "extra_pairs_backtest.csv", index=False)
    var.to_csv(tables / "extra_pairs_var.csv", index=False)
    print("\nstrategies, net of costs")
    print(strategies.set_index(["pair", "signal", "sizing"])[
        ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "p_value"]]
        .round(3).to_string())


if __name__ == "__main__":
    main()
