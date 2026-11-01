"""One set of comparison tables for every model: baselines, classical machine learning (Phase 4)
and neural networks (Phase 5).

Model forecasts are cached per job by ``scripts/run_ml.py`` (reports/predictions/ml/) and
``scripts/run_neural.py`` (reports/predictions/neural/, one file per seed). ``write_comparison``
gathers whatever is cached, averages the neural networks over their seeds, scores everything
against the baselines and writes reports/tables/ml_*.csv.

Each cache directory keeps a signature per model (its settings, the code of the module that
fits it, the model inputs and the walk-forward settings). When a signature changes, that model's
cached jobs are deleted so they run again instead of mixing old and new results.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd

from fxml.models.inputs import pair_frame
from fxml.validation.evaluate import aligned, loss, score, summary_tables
from fxml.validation.stats_tests import diebold_mariano

REFERENCE = {"volatility": "har", "returns": "zero", "direction": "frequency"}
ACTUAL = {"volatility": "y_rv", "returns": "y_ret", "direction": "y_dir"}
HEADLINE = {"volatility": "qlike", "direction": "roc_auc", "returns": "r2_oos"}


def inputs_path(cfg) -> Path:
    return cfg.paths.processed / "model_inputs.parquet"


def build_inputs(cfg) -> pd.DataFrame:
    """Every pair's modelling frame, indexed by (date, pair)."""
    table = pd.read_parquet(cfg.paths.processed / "features.parquet")
    state = pd.read_parquet(cfg.paths.processed / "market_state.parquet")
    frames = [pair_frame(table, pair, state, cfg.walk_forward.test_start).assign(pair=pair)
              for pair in cfg.pairs]
    return pd.concat(frames).set_index("pair", append=True)


def cache_dir(cfg, family: str) -> Path:
    return cfg.paths.reports / "predictions" / family


def _digest(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def signatures(specs, inputs: pd.DataFrame, cfg, *modules) -> dict[str, str]:
    """A signature for each spec's cached jobs, keyed by their file prefix "task__name"."""
    shared = _digest(pd.util.hash_pandas_object(inputs, index=True).sum(),
                     cfg.walk_forward, cfg.seed,
                     *(inspect.getsource(module) for module in modules))
    out = {}
    for spec in specs:
        settings = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)
                    if not callable(getattr(spec, f.name))}
        out[f"{spec.task}__{spec.name}"] = _digest(sorted(settings.items()), shared)
    return out


def sync_cache(directory: Path, current: dict[str, str]) -> list[str]:
    """Delete cached jobs whose model signature changed and record the new signatures.

    Jobs cached before signatures were recorded are kept and assumed current. Returns the
    prefixes whose jobs were deleted."""
    manifest = directory / "signatures.json"
    known = json.loads(manifest.read_text()) if manifest.exists() else {}
    cleared = []
    for prefix, signature in current.items():
        if prefix in known and known[prefix] != signature:
            for path in directory.glob(f"{prefix}__*"):
                path.unlink()
            cleared.append(prefix)
    manifest.write_text(json.dumps(known | current, indent=1, sort_keys=True))
    return cleared


def write_job(path: Path, forecasts: pd.DataFrame, log: pd.DataFrame) -> None:
    """Cache one job. The forecasts file appears only once complete, so an interrupted run
    never leaves a half-written file that looks finished."""
    log.to_parquet(path.with_name(path.stem + ".log.parquet"))
    forecasts.to_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)


def read_cache(directory: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(forecasts, logs) of every finished job in ``directory``."""
    if not directory.exists():
        return None, None
    files = sorted(p for p in directory.glob("*.parquet") if not p.name.endswith(".log.parquet"))
    # a log without its forecasts belongs to an interrupted job
    logs = sorted(p for p in directory.glob("*.log.parquet")
                  if p.with_name(p.name.removesuffix(".log.parquet") + ".parquet").exists())
    forecasts = pd.concat(pd.read_parquet(p) for p in files) if files else None
    return forecasts, (pd.concat(pd.read_parquet(p) for p in logs) if logs else None)


def seed_average(neural: pd.DataFrame) -> pd.DataFrame:
    """One forecast per (task, pair, model, date): the mean over seeds."""
    keys = ["task", "pair", "model", "date"]
    return neural.groupby(keys, as_index=False).agg(actual=("actual", "first"),
                                                    forecast=("forecast", "mean"))


def regime_vs_global(predictions: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano test of each regime-aware variant against its global model.

    A negative statistic means the regime-aware variant has the lower loss.
    """
    rows = []
    for (task, pair), group in predictions.groupby(["task", "pair"]):
        wide, actual = aligned(group)
        for model in wide.columns:
            for suffix in ("_regime", "_experts", "_hmm"):
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


def seed_spread(neural: pd.DataFrame, baselines: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Each seed's headline metric averaged over pairs, next to the seed average's."""
    single = neural.assign(model=neural["model"] + "_seed" + neural["seed"].astype(str))
    combined = seed_average(neural)
    references = baselines[baselines["model"].isin(reference.values())]
    scored = score(pd.concat([references, single.drop(columns="seed"), combined],
                             ignore_index=True), reference)
    rows = []
    for task, metric in HEADLINE.items():
        group = scored[(scored["task"] == task) & ~scored["model"].str.startswith("baseline_")]
        means = group.groupby("model")[metric].mean()
        for model, value in means.items():
            base, _, seed = model.partition("_seed")
            rows.append({"task": task, "model": base, "seed": seed or "average",
                         "metric": metric, "value": value})
    return pd.DataFrame(rows)


def write_comparison(cfg, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Score every cached model with the baselines and write the comparison tables."""
    baselines_file = cfg.paths.reports / "predictions" / "baselines.parquet"
    if not baselines_file.exists():
        raise SystemExit("run scripts/run_baselines.py first: the baselines' forecasts are needed")
    baselines = pd.read_parquet(baselines_file)
    baselines["model"] = "baseline_" + baselines["model"]
    reference = {task: f"baseline_{name}" for task, name in REFERENCE.items()}
    ml, ml_logs = read_cache(cache_dir(cfg, "ml"))
    neural, neural_logs = read_cache(cache_dir(cfg, "neural"))
    parts = [baselines, ml, None if neural is None else seed_average(neural)]
    predictions = pd.concat([p for p in parts if p is not None], ignore_index=True)
    predictions.to_parquet(cfg.paths.reports / "predictions" / "ml.parquet", compression="zstd")

    tables = cfg.paths.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    out = summary_tables(score(predictions, reference), reference)
    for task, table in out.items():
        table.to_csv(tables / f"ml_{task}.csv", index=False)
    if ml is not None:
        out["regime_vs_global"] = regime_vs_global(ml)
        first = ["task", "model", "pair", "year"]
        out["hyperparameters"] = ml_logs[first + [c for c in ml_logs.columns if c not in first]]
    if neural is not None:
        out["neural_seeds"] = seed_spread(neural, baselines, reference)
        first = ["task", "model", "pair", "seed", "period"]
        out["neural_training"] = neural_logs[
            first + [c for c in neural_logs.columns if c not in first]]
    for name in ("regime_vs_global", "hyperparameters", "neural_seeds", "neural_training"):
        if name in out:
            out[name].to_csv(tables / f"ml_{name}.csv", index=False)

    if verbose:
        pd.set_option("display.width", 220)
        for task in REFERENCE:
            average = out[task][out[task]["pair"] == "average"].set_index("model")
            print(f"\n{task}: average over pairs")
            print(average.drop(columns="pair").dropna(axis=1, how="all").round(4).to_string())
    return out
