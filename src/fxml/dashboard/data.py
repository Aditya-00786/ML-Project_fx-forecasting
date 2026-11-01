"""Everything the results dashboard shows, collected from the saved results into one JSON-ready
dictionary.

The static page (``scripts/build_dashboard.py``) and the Streamlit app (``dashboard/app.py``)
both read what this module writes, so they never disagree with each other or with the tables
in ``reports/tables``. Nothing here fits a model: every number comes from a saved table or
prediction file.

Each forecast is judged the way the comparison tables judge it, against a reference model with
a Diebold-Mariano test. A verdict code summarises one cell of the scorecard:

    +2  significantly better than the reference (p < 0.05)
    +1  better, not significant
    -1  worse, not significant
    -2  significantly worse
     0  the reference itself, or no test
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from fxml.events import SHORT_NAMES, STRESS_PERIODS
from fxml.validation.evaluate import aligned, loss

ALPHA = 0.05
SIGNIFICANT_DIGITS = 4

TASKS = {
    "volatility": {"label": "Volatility", "metric": "qlike", "metric_label": "QLIKE",
                   "higher_is_better": False, "reference": "baseline_har",
                   "question": "next-day realized variance"},
    "direction": {"label": "Direction", "metric": "roc_auc", "metric_label": "AUC",
                  "higher_is_better": True, "reference": "baseline_frequency",
                  "question": "whether tomorrow's return is positive"},
    "returns": {"label": "Returns", "metric": "r2_oos", "metric_label": "Out-of-sample R²",
                "higher_is_better": True, "reference": "baseline_zero",
                "question": "tomorrow's return"},
}

BASE_LABELS = {
    "baseline_naive": "Naive", "baseline_ewma": "EWMA", "baseline_garch": "GARCH",
    "baseline_gjr_garch": "GJR-GARCH", "baseline_har": "HAR-RV", "baseline_zero": "Zero",
    "baseline_mean": "Mean", "baseline_arima": "ARIMA", "baseline_frequency": "Up-day frequency",
    "baseline_persistence": "Persistence", "ridge": "Ridge", "lasso": "Lasso", "knn": "k-NN",
    "random_forest": "Random forest", "lightgbm": "LightGBM", "logistic_l2": "Logistic L2",
    "logistic_l1": "Logistic L1", "naive_bayes": "Naive Bayes", "tree": "Decision tree",
    "svm_linear": "Linear SVM", "svm_rbf": "RBF SVM", "mlp": "MLP", "lstm": "LSTM",
    "transformer": "Transformer",
}
VARIANTS = {"_regime": "GMM feature", "_experts": "GMM experts", "_hmm": "HMM feature"}
NEURAL = ("mlp", "lstm", "transformer")
GROUP_ORDER = ("Benchmarks", "Classical", "Regime-aware", "Neural")
MODEL_ORDER = list(BASE_LABELS)
HORIZON_MODELS = {  # horizons tables name models without the baseline_ prefix
    "naive": "baseline_naive", "ewma": "baseline_ewma", "garch": "baseline_garch",
    "har": "baseline_har", "frequency": "baseline_frequency",
}
STRATEGY_LABELS = {
    "flat": "Flat", "long": "Long", "carry": "Carry", "momentum": "Momentum",
    "logistic_l2": "Logistic L2", "random_forest": "Random forest", "lightgbm": "LightGBM",
    "mlp": "MLP",
}
SIZING_LABELS = {"unit": "Unit size", "vt_naive": "Vol target: naive",
                 "vt_ewma": "Vol target: EWMA", "vt_har": "Vol target: HAR-RV",
                 "vt_lightgbm": "Vol target: LightGBM"}


# ---------------------------------------------------------------- small helpers


def pair_label(pair: str) -> str:
    return f"{pair[:3]}/{pair[3:]}"


def split_variant(model: str) -> tuple[str, str | None]:
    for suffix in VARIANTS:
        if model.endswith(suffix):
            return model.removesuffix(suffix), suffix
    return model, None


def model_label(model: str) -> str:
    base, suffix = split_variant(model)
    label = BASE_LABELS.get(base, base)
    return f"{label} · {VARIANTS[suffix]}" if suffix else label


def model_group(model: str) -> str:
    base, suffix = split_variant(model)
    if base.startswith("baseline_"):
        return "Benchmarks"
    if suffix:
        return "Regime-aware"
    return "Neural" if base in NEURAL else "Classical"


def model_sort_key(model: str) -> tuple:
    base, suffix = split_variant(model)
    position = MODEL_ORDER.index(base) if base in MODEL_ORDER else len(MODEL_ORDER)
    variant = list(VARIANTS).index(suffix) if suffix else -1
    return GROUP_ORDER.index(model_group(model)), position, variant


def rounded(value, digits: int = SIGNIFICANT_DIGITS):
    """A float rounded to ``digits`` significant digits, or None for a missing value."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return round(value, digits - 1 - int(math.floor(math.log10(abs(value)))))


def verdict(stat, p) -> int:
    """Scorecard code for a Diebold-Mariano statistic (negative = beats the reference)."""
    if stat is None or p is None or not (math.isfinite(stat) and math.isfinite(p)):
        return 0
    return (2 if p < ALPHA else 1) * (1 if stat < 0 else -1)


def records(frame: pd.DataFrame) -> list[dict]:
    """Rows as dictionaries with floats rounded and missing values as None."""
    out = []
    for row in frame.to_dict("records"):
        out.append({k: (rounded(v) if isinstance(v, float | np.floating) else
                        (int(v) if isinstance(v, np.integer) else v)) for k, v in row.items()})
    return out


def _dm_columns(table: pd.DataFrame) -> tuple[str, str]:
    stat = next(c for c in table.columns if c.startswith("dm_vs_"))
    return stat, next(c for c in table.columns if c.startswith("dm_p_vs_"))


# ---------------------------------------------------------------- scorecards


def scorecard(table: pd.DataFrame, task: str, pairs: list[str]) -> dict:
    """One task's scorecard: columns (models), cells (pair x model) and a summary per model."""
    spec = TASKS[task]
    stat_col, p_col = _dm_columns(table)
    rows = table[table["pair"].isin(pairs)]
    models = sorted(rows["model"].unique(), key=model_sort_key)
    cells = []
    for row in rows.itertuples(index=False):
        row = row._asdict()
        stat, p = row[stat_col], row[p_col]
        cell = {"pair": row["pair"], "model": row["model"], "value": rounded(row[spec["metric"]]),
                "dm": rounded(stat), "p": rounded(p), "v": verdict(stat, p),
                "days": int(row["days"]) if pd.notna(row["days"]) else None}
        if "pt_p" in row:
            cell["pt_p"] = rounded(row["pt_p"])
        cells.append(cell)
    summary = []
    for model in models:
        mine = [c for c in cells if c["model"] == model]
        values = [c["value"] for c in mine if c["value"] is not None]
        summary.append({"model": model,
                        "better": sum(c["v"] == 2 for c in mine),
                        "worse": sum(c["v"] == -2 for c in mine),
                        "mean": rounded(np.mean(values)) if values else None})
    columns = [{"id": m, "label": model_label(m), "group": model_group(m)} for m in models]
    return {"columns": columns, "cells": cells, "summary": summary}


def cumulative_advantage(predictions: pd.DataFrame, task: str, months: pd.PeriodIndex) -> dict:
    """Per pair and model, the running total of the reference's loss minus the model's loss,
    summed by month: a line that climbs is a model pulling ahead of the reference."""
    reference = TASKS[task]["reference"]
    out = {}
    for pair, group in predictions[predictions["task"] == task].groupby("pair"):
        wide, actual = aligned(group)
        if reference not in wide:
            continue
        base = pd.Series(loss(task, actual, wide[reference]), index=wide.index)
        per_pair = {}
        for model in wide.columns:
            if model == reference:
                continue
            gap = (base - pd.Series(loss(task, actual, wide[model]), index=wide.index)).dropna()
            if gap.empty:
                continue
            monthly = gap.groupby(gap.index.to_period("M")).sum().reindex(months)
            path = monthly.fillna(0).cumsum().where(monthly.notna().cumsum() > 0)
            per_pair[model] = [rounded(v) for v in path]
        out[pair] = per_pair
    return out


def test_months(*frames: pd.DataFrame) -> pd.PeriodIndex:
    dates = pd.concat([pd.Series(f["date"].unique()) for f in frames])
    return pd.period_range(pd.Timestamp(dates.min()), pd.Timestamp(dates.max()), freq="M")


def horizons(tables: Path, ml: dict[str, pd.DataFrame]) -> dict:
    """Scorecards by forecast horizon: one day (the main comparison) against 5 and 22 days."""
    out = {}
    for task in ("volatility", "direction"):
        longer = pd.read_csv(tables / f"horizons_{task}.csv")
        longer["model"] = longer["model"].map(lambda m: HORIZON_MODELS.get(m, m))
        stat_col, p_col = _dm_columns(longer)
        one = ml[task][ml[task]["model"].isin(longer["model"].unique())].assign(horizon=1)
        one_stat, one_p = _dm_columns(one)
        one = one.rename(columns={one_stat: stat_col, one_p: p_col})
        both = pd.concat([one, longer], ignore_index=True)
        models = sorted(longer["model"].unique(), key=model_sort_key)
        cells = []
        for row in both.itertuples(index=False):
            row = row._asdict()
            cells.append({"h": int(row["horizon"]), "pair": row["pair"], "model": row["model"],
                          "value": rounded(row[TASKS[task]["metric"]]),
                          "dm": rounded(row[stat_col]), "p": rounded(row[p_col]),
                          "v": verdict(row[stat_col], row[p_col])})
        out[task] = {"columns": [{"id": m, "label": model_label(m), "group": model_group(m)}
                                 for m in models],
                     "horizons": [1, 5, 22], "cells": cells}
    return out


# ---------------------------------------------------------------- research questions


def research_questions(tables: Path, ml: dict[str, pd.DataFrame], pairs: list[str]) -> list[dict]:
    """The five research questions, each answered from the tables with its test and source."""
    n = len(pairs)

    def average(task: str) -> pd.DataFrame:
        return ml[task][ml[task]["pair"] == "average"].set_index("model")

    vol = average("volatility")
    learners = vol[[not m.startswith("baseline_") for m in vol.index]]
    best = learners["qlike"].idxmin()
    gain = 1 - vol.loc[best, "qlike"] / vol.loc["baseline_har", "qlike"]
    garch = vol.loc["baseline_garch", "qlike"] / vol.loc["baseline_har", "qlike"] - 1
    hz = pd.read_csv(tables / "horizons_volatility.csv")
    hz_avg = hz[hz["pair"] == "average"].set_index(["horizon", "model"])["qlike"]
    hz_gap = {h: hz_avg[(h, "lightgbm")] / hz_avg[(h, "har")] - 1 for h in (5, 22)}

    direction = average("direction")
    dir_learners = direction[[not m.startswith("baseline_") for m in direction.index]]
    best_auc = dir_learners["roc_auc"].idxmax()
    pt = ml["direction"][ml["direction"]["pair"].isin(pairs)]
    pt_counts = (pt["pt_p"] < ALPHA).groupby(pt["model"]).sum()
    returns = average("returns")
    best_r2 = returns.loc[[m for m in returns.index if m != "baseline_zero"], "r2_oos"].max()

    variants = pd.read_csv(tables / "ml_regime_vs_global.csv")
    variants = variants[variants["pair"].isin(pairs)]
    sig = variants["dm_p_vs_global"] < ALPHA
    regime_better = int((sig & (variants["dm_vs_global"] < 0)).sum())
    regime_worse = int((sig & (variants["dm_vs_global"] > 0)).sum())
    summary = pd.read_csv(tables / "regimes_summary.csv")
    gmm = summary[(summary["method"] == "gmm") & (summary["labels"] == "walk_forward")]
    gmm = gmm.set_index("regime")

    strategies = pd.read_csv(tables / "backtest_strategies.csv")
    portfolio = strategies[strategies["pair"] == "portfolio"].set_index(["signal", "sizing"])
    carry = portfolio.loc[("carry", "unit")]
    signals = portfolio.loc[[s for s in portfolio.index.get_level_values(0).unique()
                             if s in ("logistic_l2", "random_forest", "lightgbm", "mlp")]]
    signal_sharpes = signals.xs("unit", level="sizing")["sharpe"]
    var = pd.read_csv(tables / "risk_var.csv")
    normal = var[var["method"] == "normal"]
    kupiec = (normal["kupiec_p"] >= ALPHA).groupby(normal["model"]).mean()

    curves = pd.read_csv(tables / "ml_learning_curves.csv")
    hv = curves[(curves["target"] == "high_volatility") & (curves["size"] > 0)]
    ece = hv.pivot(index="size", columns="model", values="ece")
    hv = hv.pivot(index="size", columns="model", values="roc_auc")
    nb_ahead = hv.index[hv["naive_bayes"] > hv["logistic"]]
    logistic_ahead = hv.index[hv["logistic"] >= hv["naive_bayes"]]

    # Every verdict word is decided by the tables, not written in by hand.
    n_best = int(vol.loc[best, "pairs_significantly_better"])
    vol_wins = gain > 0 and n_best > n / 2
    rq1 = ("Yes, at one day" if vol_wins and min(hz_gap.values()) > 0 else
           "Yes" if vol_wins else "No")
    rq2 = "No" if pt_counts.max() <= 1 and best_r2 <= 0 else "Partly"
    calm_vol, turbulent_vol = gmm.loc[0, "next_day_vol_pct"], gmm.loc[1, "next_day_vol_pct"]
    rq3 = ("Regimes yes, " + ("models no" if regime_better <= regime_worse else "models yes")
           if turbulent_vol > calm_vol else "No")
    signal_p = signals.xs("unit", level="sizing")["p_value"]
    rq4 = ("No for trading" if signal_sharpes.max() <= 0 or signal_p.min() >= ALPHA
           else "Yes")
    generative_first = (len(nb_ahead) > 0 and len(logistic_ahead) > 0
                        and nb_ahead.max() < logistic_ahead.min())
    rq5 = ("Generative first, then discriminative" if generative_first else
           "Discriminative throughout" if len(nb_ahead) == 0 else "Mixed")
    nb_worse_calibrated = bool((ece["naive_bayes"] > ece["logistic"]).all())
    # Slide headlines follow the same verdicts, so a rerun can never leave them contradicting
    headlines = {
        "RQ1": ("Learning beats HAR-RV on one-day volatility" if vol_wins
                else "Learning does not beat HAR-RV on volatility"),
        "RQ2": ("Direction and returns: no better than chance" if rq2 == "No"
                else "Some direction is predictable"),
        "RQ3": ("Calm and turbulent regimes, but no better forecasts"
                if rq3 == "Regimes yes, models no"
                else "Regimes that improve the forecasts" if rq3.startswith("Regimes")
                else "No clear regimes"),
        "RQ4": ("Nothing pays after costs" if rq4 == "No for trading"
                else "Some signals pay after costs"),
        "RQ5": ("Naive Bayes learns faster, logistic regression ends better" if generative_first
                else "Logistic regression leads at every size" if len(nb_ahead) == 0
                else "No clear winner as the data grows"),
    }
    horizon_headline = ("The edge is a one-day effect" if rq1 == "Yes, at one day"
                        else "The edge holds at longer horizons" if vol_wins
                        else "HAR-RV holds at every horizon")

    return [
        {"id": "RQ1", "question": "How much of next-day realized volatility can be predicted, "
                                   "and do machine learning models improve on GARCH and HAR-RV?",
         "verdict": rq1,
         "short": f"{model_label(best)} {gain:.1%} below HAR-RV; {n_best} of {n} pairs",
         "answer": f"{model_label(best)}'s QLIKE is {gain:.1%} below HAR-RV's on average, "
                   f"significantly lower on {n_best} of {n} pairs. GARCH is {garch:.0%} worse "
                   f"than HAR-RV. At 5 and 22 days LightGBM is {hz_gap[5]:.0%} and "
                   f"{hz_gap[22]:.0%} " + ("worse" if min(hz_gap.values()) > 0 else "different")
                   + " than HAR-RV.",
         "evidence": [{"label": "QLIKE vs HAR-RV", "test": "Diebold–Mariano, 5%",
                       "source": "reports/tables/ml_volatility.csv"},
                      {"label": "5 and 22 days", "test": "Diebold–Mariano with overlap",
                       "source": "reports/tables/horizons_volatility.csv"}],
         "view": {"task": "volatility"}, "headline": headlines["RQ1"],
         "horizon_headline": horizon_headline},
        {"id": "RQ2", "question": "Is the direction of next-day returns predictable beyond "
                                   "chance once look-ahead bias is removed?",
         "verdict": rq2,
         "short": f"best AUC {dir_learners.loc[best_auc, 'roc_auc']:.3f}; calls beat chance "
                  f"on at most {int(pt_counts.max())} of {n} pairs",
         "answer": f"The best average AUC is {dir_learners.loc[best_auc, 'roc_auc']:.3f} "
                   f"({model_label(best_auc)}), against 0.5 for a coin. No model's up/down calls "
                   f"beat chance on more than {int(pt_counts.max())} of {n} pairs, and no return "
                   f"forecast beats a zero forecast (best out-of-sample R² {best_r2:.4f}).",
         "evidence": [{"label": "Up/down calls", "test": "Pesaran–Timmermann, 5%",
                       "source": "reports/tables/ml_direction.csv"},
                      {"label": "Return forecasts", "test": "Diebold–Mariano vs zero",
                       "source": "reports/tables/ml_returns.csv"}],
         "view": {"task": "direction"}, "headline": headlines["RQ2"]},
        {"id": "RQ3", "question": "Do unsupervised methods recover economically meaningful "
                                   "regimes, and do regime-aware models do better?",
         "verdict": rq3,
         "short": f"regimes help in {regime_better} comparisons, hurt in {regime_worse}",
         "answer": f"A calm and a turbulent regime emerge: next-day volatility averages "
                   f"{calm_vol:.1f}% a year in the calm one and "
                   f"{turbulent_vol:.1f}% in the turbulent one. Adding regimes "
                   f"to a model helps significantly in {regime_better} pair-model comparisons "
                   f"and hurts in {regime_worse}.",
         "evidence": [{"label": "Regime profiles", "test": "walk-forward labels",
                       "source": "reports/tables/regimes_summary.csv"},
                      {"label": "Regime-aware vs global", "test": "Diebold–Mariano, 5%",
                       "source": "reports/tables/ml_regime_vs_global.csv"}],
         "view": {"task": "volatility", "group": "Regime-aware"},
         "headline": headlines["RQ3"]},
        {"id": "RQ4", "question": "Do forecast gains survive transaction costs, and give better "
                                   "risk-adjusted returns and more reliable risk estimates?",
         "verdict": rq4,
         "short": f"direction signals' Sharpe {signal_sharpes.min():.2f} to "
                  f"{signal_sharpes.max():.2f}; carry {carry['sharpe']:.2f}",
         "answer": ("No direction signal makes money after costs" if signal_sharpes.max() <= 0
                    else "No direction signal's profit is significant after costs")
                   + f": their Sharpe ratios run from "
                   f"{signal_sharpes.min():.2f} to {signal_sharpes.max():.2f}. Carry earns "
                   f"{carry['sharpe']:.2f} (p = {carry['p_value']:.2f}). Better volatility "
                   f"forecasts do help risk: normal Value at Risk from LightGBM passes Kupiec's "
                   f"test on {kupiec.get('lightgbm', np.nan):.0%} of pair-sides, against "
                   f"{kupiec.get('baseline_ewma', np.nan):.0%} for EWMA.",
         "evidence": [{"label": "Equal-risk portfolio", "test": "Newey–West t-test",
                       "source": "reports/tables/backtest_strategies.csv"},
                      {"label": "99% Value at Risk", "test": "Kupiec and Christoffersen",
                       "source": "reports/tables/risk_var.csv"}],
         "view": {"section": "backtest"}, "headline": headlines["RQ4"]},
        {"id": "RQ5", "question": "How does a generative classifier compare with a "
                                   "discriminative one as the training data grows?",
         "verdict": rq5,
         "short": (f"Naive Bayes ahead up to {int(nb_ahead.max())} days, logistic from "
                   f"{int(logistic_ahead.min())}" if generative_first else
                   "logistic regression ahead at every size"),
         "answer": (f"On the high-volatility target Naive Bayes ranks days better with "
                    f"{int(nb_ahead.max())} training days or fewer; logistic regression catches "
                    f"up by {int(logistic_ahead.min())} days and stays ahead."
                    if generative_first else
                    "On the high-volatility target the two rank days differently at "
                    "different sizes.")
                   + (" Naive Bayes is the worse calibrated of the two at every size."
                      if nb_worse_calibrated else ""),
         "evidence": [{"label": "Learning curves", "test": "pooled AUC and calibration error",
                       "source": "reports/tables/ml_learning_curves.csv"}],
         "view": {"section": "learning"}, "headline": headlines["RQ5"]},
    ]


# ---------------------------------------------------------------- other sections


def regime_timeline(state: pd.DataFrame, first_refit) -> dict:
    """Weekly turbulent-regime probabilities from the mixture and the hidden Markov model."""
    weekly = state[["gmm_p1", "hmm_p1"]].resample("W-FRI").last().dropna(how="all")
    return {"dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
            "gmm": [rounded(v, 3) for v in weekly["gmm_p1"]],
            "hmm": [rounded(v, 3) for v in weekly["hmm_p1"]],
            "first_refit": pd.Timestamp(first_refit).strftime("%Y-%m-%d")}


def backtest(tables: Path, predictions: Path) -> dict:
    strategies = pd.read_csv(tables / "backtest_strategies.csv")
    daily = pd.read_parquet(predictions / "backtest.parquet")
    daily = daily[daily["pair"] == "portfolio"]
    months = pd.period_range(daily["date"].min(), daily["date"].max(), freq="M")
    curves = {}
    for (signal, sizing), group in daily.groupby(["signal", "sizing"]):
        if signal == "flat":
            continue
        monthly = group.groupby(group["date"].dt.to_period("M"))["net"].sum().reindex(months)
        curves[f"{signal}|{sizing}"] = [rounded(v) for v in monthly.fillna(0).cumsum()]
    breakdown = pd.read_csv(tables / "backtest_breakdown.csv")
    breakdown = breakdown[breakdown["sizing"].isin(["unit", "vt_lightgbm"])]
    targeting = pd.read_csv(tables / "backtest_risk_targeting.csv")
    targeting = targeting.groupby("sizing", as_index=False)[
        ["mean_realised_vol", "tracking", "quarters_above_1_5x"]].mean()
    var = pd.read_csv(tables / "risk_var.csv")
    var_summary = var.groupby(["model", "method", "side"], as_index=False).agg(
        rate=("rate", "mean"), kupiec_pass=("kupiec_p", lambda p: int((p >= ALPHA).sum())),
        christoffersen_pass=("christoffersen_p", lambda p: int((p >= ALPHA).sum())),
        pairs=("pair", "nunique"))
    return {
        "months": [str(m) for m in months],
        "curves": curves,
        "portfolio": records(strategies[strategies["pair"] == "portfolio"].drop(columns="pair")),
        "per_pair": records(strategies[strategies["pair"] != "portfolio"][
            ["signal", "sizing", "pair", "sharpe", "ann_return", "ann_vol", "p_value"]]),
        "breakdown": records(breakdown[["dimension", "bucket", "signal", "sizing", "days",
                                        "sharpe", "ann_return", "ann_vol", "max_drawdown"]]),
        "targeting": records(targeting),
        "var": records(var_summary),
        "strategy_labels": STRATEGY_LABELS,
        "sizing_labels": SIZING_LABELS,
    }


def learning(tables: Path) -> dict:
    curves = pd.read_csv(tables / "ml_learning_curves.csv")
    return {"rows": records(curves[["target", "model", "size", "train_days", "accuracy",
                                    "log_loss", "brier", "roc_auc", "ece"]])}


def calibration(ml: dict[str, pd.DataFrame]) -> list[dict]:
    direction = ml["direction"]
    rows = direction[direction["pair"] == "average"][
        ["model", "roc_auc", "brier", "ece", "calibration_slope", "calibration_slope_se"]]
    return records(rows.assign(label=rows["model"].map(model_label)))


def shap(tables: Path, top: int = 12) -> dict:
    values = pd.read_csv(tables / "shap_lightgbm.csv")
    out = {}
    for task, group in values.groupby("task"):
        mean = (group.groupby(["feature", "group"], as_index=False)["mean_abs_shap"].mean()
                .sort_values("mean_abs_shap", ascending=False).head(top))
        out[task] = records(mean)
    return out


def data_quality(tables: Path, pairs: list[str], extra_pairs: list[str]) -> dict:
    coverage = pd.read_csv(tables / "data_coverage.csv", index_col=0)
    h10 = pd.read_csv(tables / "data_h10_check.csv", index_col=0)
    core = coverage.loc[pairs].join(h10[["return_corr", "return_corr_clock_gap",
                                         "mean_abs_level_gap_pct"]])
    rows = core.assign(kind="core")
    extra_file = tables / "extra_data_check.csv"
    if extra_file.exists():
        extra = pd.read_csv(extra_file, index_col=0).loc[extra_pairs]
        rows = pd.concat([rows, extra[rows.columns.drop("kind")].assign(kind="extra")])
    rows = rows.rename_axis("pair").reset_index()
    keep = ["pair", "kind", "first", "last", "coverage_pct", "partial_days", "return_corr",
            "return_corr_clock_gap", "mean_abs_level_gap_pct"]
    return {"rows": records(rows[keep])}


def extra_pairs(tables: Path, predictions: Path, extra: list[str], daily_only: dict) -> dict | None:
    if not (tables / "extra_pairs_volatility.csv").exists():
        return None
    preds = pd.read_parquet(predictions / "extra_pairs.parquet")
    preds = preds[preds["actual"].notna()]
    months = test_months(preds)
    cards, traces = {}, {}
    for task in TASKS:
        table = pd.read_csv(tables / f"extra_pairs_{task}.csv")
        cards[task] = scorecard(table, task, [p for p in extra if p in set(table["pair"])])
        traces[task] = cumulative_advantage(preds, task, months)
    strategies = pd.read_csv(tables / "extra_pairs_backtest.csv")
    strategies = strategies[strategies["sizing"].isin(["unit", "vt_ewma"])]
    return {"pairs": extra, "daily_only": sorted(daily_only), "months": [str(m) for m in months],
            "scorecards": cards, "traces": traces,
            "summary": extra_summary(tables, extra, daily_only, strategies),
            "backtest": records(strategies[["pair", "signal", "sizing", "sharpe", "sortino",
                                            "ann_return", "ann_vol", "max_drawdown",
                                            "p_value"]])}


def extra_summary(tables: Path, extra: list[str], daily_only: dict,
                  strategies: pd.DataFrame) -> dict:
    """What the extra pairs show, in sentences built from their tables."""
    pl = {p: pair_label(p) for p in extra}
    vol = pd.read_csv(tables / "extra_pairs_volatility.csv")
    sentences = []
    for pair in [p for p in extra if p in set(vol["pair"])]:
        rows = vol[vol["pair"] == pair].set_index("model")
        learners = rows[[not m.startswith("baseline_") for m in rows.index]]
        best = learners["qlike"].idxmin()
        gain = 1 - rows.loc[best, "qlike"] / rows.loc["baseline_har", "qlike"]
        p = rows.loc[best, "dm_p_vs_baseline_har"]
        verdict = "significantly" if p < ALPHA and gain > 0 else "not significantly"
        sentences.append(f"{pl[pair]} volatility: the best learning model ({model_label(best)}) is "
                         f"{abs(gain):.1%} {'below' if gain > 0 else 'above'} HAR-RV, "
                         f"{verdict} (p = {p:.2f}).")
    direction = pd.read_csv(tables / "extra_pairs_direction.csv")
    beating = {}
    for pair in extra:
        rows = direction[(direction["pair"] == pair) & (direction["pt_p"] < ALPHA)
                         & ~direction["model"].str.startswith("baseline_")]
        beating[pair] = rows
        if rows.empty:
            sentences.append(f"{pl[pair]} direction: no model's up/down calls beat chance.")
        else:
            named = " and ".join(f"{model_label(r.model)} (AUC {r.roc_auc:.3f})"
                                 for r in rows.itertuples())
            sentences.append(f"{pl[pair]} direction: {named} beat chance on up/down calls "
                             "(Pesaran–Timmermann, 5%).")
    trading = strategies[strategies["signal"] != "flat"]
    best = trading.loc[trading["sharpe"].idxmax()]
    significant = trading[(trading["p_value"] < ALPHA) & (trading["sharpe"] > 0)]
    if significant.empty:
        sentences.append(f"No strategy's profit is significant after costs (best Sharpe "
                         f"{best['sharpe']:.2f}, p = {best['p_value']:.2f}).")
    else:
        sentences.append(f"{len(significant)} strategies earn a significant profit after costs.")
    caveat = any(not beating[p].empty for p in daily_only)
    return {"sentences": sentences, "daily_only_beats_chance": caveat,
            "headline": " ".join(sentences)}


# ---------------------------------------------------------------- everything


def collect(cfg) -> dict:
    """Every number the dashboard shows, from ``reports/`` and the walk-forward market state."""
    tables = cfg.paths.reports / "tables"
    predictions = cfg.paths.reports / "predictions"
    pairs = list(cfg.pairs)
    ml = {task: pd.read_csv(tables / f"ml_{task}.csv") for task in TASKS}
    forecasts = pd.read_parquet(predictions / "ml.parquet")
    months = test_months(forecasts)
    state = pd.read_parquet(cfg.paths.processed / "market_state.parquet")
    first_refit = pd.Timestamp(cfg.walk_forward.test_start)
    return {
        "meta": {
            "title": "Regime-aware learning of returns and volatility in G10 FX",
            "course": "BITS F464 Machine Learning",
            "test_start": str(months[0]), "test_end": str(months[-1]),
            "data_to": pd.Timestamp(forecasts["date"].max()).strftime("%Y-%m-%d"),
            "pairs": pairs, "pair_labels": {p: pair_label(p)
                                            for p in [*pairs, *cfg.extra_pairs]},
            "extra_pairs": list(cfg.extra_pairs),
            "refit": "monthly (neural networks quarterly)",
            "tasks": TASKS,
            "alpha": ALPHA,
            "stress": [{"name": name, "short": SHORT_NAMES[name], "start": start, "end": end}
                       for name, (start, end) in STRESS_PERIODS.items()],
        },
        "questions": research_questions(tables, ml, pairs),
        "scorecards": {task: scorecard(ml[task], task, pairs) for task in TASKS},
        "months": [str(m) for m in months],
        "traces": {task: cumulative_advantage(forecasts, task, months) for task in TASKS},
        "horizons": horizons(tables, ml),
        "regimes": {"timeline": regime_timeline(state, first_refit),
                    "summary": records(pd.read_csv(tables / "regimes_summary.csv").query(
                        "method in ['gmm', 'hmm'] and labels == 'walk_forward'")[
                        ["method", "regime", "share_of_days", "mean_spell_days",
                         "stay_probability", "next_day_vol_pct", "stress_capture"]])},
        "backtest": backtest(tables, predictions),
        "learning": learning(tables),
        "calibration": calibration(ml),
        "shap": shap(tables),
        "data": data_quality(tables, pairs, list(cfg.extra_pairs)),
        "extra": extra_pairs(tables, predictions, list(cfg.extra_pairs), cfg.daily_only),
    }
