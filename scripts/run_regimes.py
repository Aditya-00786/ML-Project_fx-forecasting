"""Find market regimes and currency factors.

1. Builds the daily market state from the feature table.
2. Chooses one number of regimes for every method on the first training window only (before
   the walk-forward test start), so the choice never sees test data.
3. Labels every day with models refitted once a year on earlier days only (walk-forward), and
   also fits each model once on the full sample to describe the history. A hidden Markov model
   with the same number of regimes adds filtered (forward-only) regime probabilities.
4. Does the same yearly walk-forward refitting for the currency factors.

Writes data/processed/market_state.parquet and reports/tables/regimes_*.csv and factors_*.csv.

    uv run python scripts/run_regimes.py
"""

from __future__ import annotations

import pandas as pd

from fxml.config import load_config
from fxml.events import stress_label
from fxml.factors import CurrencyFactors, usd_value_returns, walk_forward_factors
from fxml.regimes.evaluate import (
    agreement,
    regime_profile,
    selection_table,
    transition_matrix,
    vote_k,
)
from fxml.regimes.hmm import HMMRegimes, walk_forward_hmm
from fxml.regimes.models import RegimeModel, walk_forward_labels
from fxml.regimes.state import market_state

METHODS = ("kmeans", "ward", "gmm", "dbscan")
ALL_METHODS = (*METHODS, "hmm")  # the hidden Markov model is fitted separately
K_RANGE = range(2, 7)
REFIT = "YS"  # regimes describe slow-moving conditions, so a yearly refit is enough
MIN_SHARE = 0.10  # smallest regime must hold this share of training days


def main() -> None:
    cfg = load_config()
    table = pd.read_parquet(cfg.paths.processed / "features.parquet")
    pairs = cfg.pairs
    first_refit = pd.Timestamp(cfg.walk_forward.test_start)

    state = market_state(table, pairs)
    complete = state.dropna()
    print(f"market state: {len(complete):,} complete days, "
          f"{complete.index[0]:%Y-%m-%d} to {complete.index[-1]:%Y-%m-%d}")

    selection = selection_table(complete[complete.index < first_refit], K_RANGE, seed=cfg.seed)
    k, votes = vote_k(selection, MIN_SHARE)
    selection["votes"] = selection.apply(
        lambda r: ((votes["method"] == r["method"]) & (votes["k"] == r["k"])).sum(), axis=1)
    chosen = dict.fromkeys(("kmeans", "ward", "gmm"), k) | {"dbscan": None}
    print(f"number of regimes: {k} ({(votes['k'] == k).sum()} of {len(votes)} votes)")

    out = state.copy()
    for method in METHODS:
        labels = walk_forward_labels(complete, method, chosen[method], first_refit, REFIT,
                                     cfg.seed)
        out[f"regime_{method}"] = labels["regime"]
        for column in [c for c in labels.columns if c.startswith("p")]:
            out[f"{method}_{column}"] = labels[column]
        full = RegimeModel(method, chosen[method], seed=cfg.seed).fit(complete)
        out[f"full_{method}"] = pd.Series(full.labels_, index=complete.index)

    # Hidden Markov model with the same number of regimes, filtered probabilities only
    hmm = walk_forward_hmm(complete, k, first_refit, REFIT, cfg.seed)
    out["regime_hmm"] = hmm["regime"]
    for column in [c for c in hmm.columns if c.startswith("hmm_p")]:
        out[column] = hmm[column]
    full_hmm = HMMRegimes(k, cfg.seed).fit(complete)
    out["full_hmm"] = pd.Series(full_hmm.filter(complete).argmax(axis=1), index=complete.index)
    training = complete[complete.index < first_refit]
    hmm_selection = pd.DataFrame(
        [{"k": n, "bic": HMMRegimes(n, cfg.seed).fit(training).bic(training)} for n in K_RANGE])
    out["labels_in_sample"] = out.index < first_refit

    returns = usd_value_returns(table, pairs)
    factor_returns, loadings = walk_forward_factors(returns, first_refit, REFIT)
    out = out.join(factor_returns.drop(columns="in_sample"))
    full_factors = CurrencyFactors(3).fit(returns)
    loadings = pd.concat([
        full_factors.loadings_.rename_axis("currency").reset_index().assign(fit="full sample"),
        loadings.assign(fit=loadings["fit"].dt.strftime("%Y-%m-%d")),
    ], ignore_index=True)
    explained = full_factors.explained_variance_ratio_

    path = cfg.paths.processed / "market_state.parquet"
    out.to_parquet(path, compression="zstd")
    print(f"wrote {path} with {out.shape[1]} columns")

    # Profiles: next-day volatility averaged over pairs, and the documented stress episodes.
    next_rv = table["y_rv"].unstack("pair")[list(pairs)].mean(axis=1)
    stress = stress_label(out.index) != "Normal"
    profiles, transitions = [], []
    test = out[out.index >= first_refit]
    for method in ALL_METHODS:
        for kind, column, rows in [("walk_forward", f"regime_{method}", test),
                                   ("full_sample", f"full_{method}", out)]:
            labels = rows[column].dropna().astype(int)
            profile = regime_profile(labels, next_rv, stress, state)
            profiles.append(profile.assign(method=method, labels=kind))
            moves = transition_matrix(labels).stack().rename("probability").reset_index()
            transitions.append(moves.assign(method=method, labels=kind))
    profiles = pd.concat(profiles, ignore_index=True)
    first = ["method", "labels", "regime"]
    profiles = profiles[first + [c for c in profiles.columns if c not in first]]
    transitions = pd.concat(transitions, ignore_index=True)[
        ["method", "labels", "from", "to", "probability"]]
    walk_forward = test[[f"regime_{m}" for m in ALL_METHODS]]
    ari = agreement(walk_forward.set_axis(ALL_METHODS, axis=1))

    tables = cfg.paths.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    selection.to_csv(tables / "regimes_selection.csv", index=False)
    profiles.to_csv(tables / "regimes_summary.csv", index=False)
    transitions.to_csv(tables / "regimes_transitions.csv", index=False)
    ari.to_csv(tables / "regimes_agreement.csv")
    hmm_selection.to_csv(tables / "regimes_hmm_selection.csv", index=False)
    full_hmm.transition_matrix().stack().rename("probability").reset_index().to_csv(
        tables / "regimes_hmm_transitions.csv", index=False)
    loadings.to_csv(tables / "factors_loadings.csv", index=False)
    explained.rename("explained_variance").to_csv(tables / "factors_explained.csv",
                                                   index_label="component")

    pd.set_option("display.width", 200)
    print("\nselection on the first training window")
    print(selection.round(3).to_string(index=False))
    print("\nwalk-forward regimes from", f"{first_refit:%Y-%m-%d}")
    print(profiles[profiles["labels"] == "walk_forward"].drop(columns="labels")
          .round(3).to_string(index=False))
    print("\nagreement between methods (adjusted Rand index)")
    print(ari.round(2).to_string())
    print("\nhidden Markov model: BIC on the first training window, and the full-sample "
          "transition matrix")
    print(hmm_selection.round(1).to_string(index=False))
    print(full_hmm.transition_matrix().round(3).to_string())
    print("\nfull-sample factor loadings")
    print(full_factors.loadings_.round(2).to_string())
    print(explained.round(3).to_string())


if __name__ == "__main__":
    main()
