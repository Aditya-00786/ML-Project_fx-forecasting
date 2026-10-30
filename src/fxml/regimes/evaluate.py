"""Choosing the number of regimes, and describing the regimes a model finds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from fxml.regimes.models import RegimeModel


def selection_table(
    state: pd.DataFrame, ks=range(2, 7), methods=("kmeans", "ward", "gmm"), seed: int = 42
) -> pd.DataFrame:
    """Silhouette, Davies–Bouldin and (Gaussian mixture only) BIC for each method and k.

    Higher silhouette and lower Davies–Bouldin mean tighter, better separated clusters; lower
    BIC means a better mixture after penalising its parameters. ``min_share`` is the smallest
    regime's share of days, which exposes solutions that only split off a handful of outliers.
    """
    state = state.dropna()
    z = StandardScaler().fit_transform(state)
    rows = []
    for method in methods:
        for k in ks:
            model = RegimeModel(method, k, seed=seed).fit(state)
            labels = model.labels_
            rows.append({
                "method": method, "k": k,
                "silhouette": silhouette_score(z, labels),
                "davies_bouldin": davies_bouldin_score(z, labels),
                "bic": model.model_.bic(z) if method == "gmm" else np.nan,
                "min_share": np.bincount(labels, minlength=k).min() / len(labels),
            })
    return pd.DataFrame(rows)


CRITERIA = {"silhouette": "max", "davies_bouldin": "min", "bic": "min"}


def vote_k(selection: pd.DataFrame, min_share: float = 0.10) -> tuple[int, pd.DataFrame]:
    """One number of regimes shared by every method, so their regimes can be compared.

    Each method and criterion votes for its best k among solutions whose smallest regime holds
    at least ``min_share`` of the days, so that every regime has enough history to train a
    model on. The most voted k wins, ties going to the smaller k. Returns the k and the votes.
    """
    allowed = selection[selection["min_share"] >= min_share]
    votes = []
    for method, rows in allowed.groupby("method", sort=False):
        for criterion, best in CRITERIA.items():
            scores = rows.set_index("k")[criterion].dropna()
            if len(scores):
                k = scores.idxmax() if best == "max" else scores.idxmin()
                votes.append({"method": method, "criterion": criterion, "k": int(k)})
    if not votes:
        raise ValueError(f"no solution keeps every regime above {min_share:.0%} of days")
    votes = pd.DataFrame(votes)
    counts = votes["k"].value_counts()
    return int(counts[counts == counts.max()].index.min()), votes


def spells(labels: pd.Series) -> pd.DataFrame:
    """Unbroken runs of one regime: regime, first day and length in trading days."""
    labels = labels.dropna()
    new_spell = labels.ne(labels.shift()).cumsum()
    grouped = labels.groupby(new_spell)
    return pd.DataFrame({
        "regime": grouped.first().to_numpy(),
        "start": grouped.apply(lambda s: s.index[0]).to_numpy(),
        "length": grouped.size().to_numpy(),
    })


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    """Share of days in regime i (rows) followed by regime j (columns) on the next trading day."""
    labels = labels.dropna()
    counts = pd.crosstab(labels.iloc[:-1].to_numpy(), labels.iloc[1:].to_numpy())
    counts.index.name, counts.columns.name = "from", "to"
    return counts.div(counts.sum(axis=1), axis=0)


def regime_profile(
    labels: pd.Series, next_rv: pd.Series, stress: pd.Series, state: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per regime: how common and persistent it is, and what follows it.

    ``next_rv`` is the next day's realized variance averaged over pairs, reported as annualised
    volatility in percent, and ``stress`` flags days inside a documented stress episode.
    ``stress_capture`` is the share of all stress days that fall in the regime. If ``state`` is
    given, its average on the regime's days is added to describe what the regime looks like.
    """
    labels = labels.dropna().astype(int)
    runs = spells(labels)
    stress = stress.reindex(labels.index).fillna(False).astype(bool)
    moves = transition_matrix(labels)
    rows = []
    for regime in sorted(labels.unique()):
        days = labels == regime
        length = runs.loc[runs["regime"] == regime, "length"]
        known = regime in moves.index and regime in moves.columns
        stay = moves.at[regime, regime] if known else np.nan  # never followed by another day
        rows.append({
            "regime": regime,
            "share_of_days": days.mean(),
            "spells": len(length),
            "mean_spell_days": length.mean(),
            "median_spell_days": length.median(),
            "stay_probability": stay,
            "next_day_vol_pct": 100 * np.sqrt(next_rv.reindex(labels.index)[days].mean() * 252),
            "stress_share": stress[days].mean(),
            "stress_capture": stress[days].sum() / max(stress.sum(), 1),
        })
    profile = pd.DataFrame(rows)
    if state is not None:
        means = state.reindex(labels.index).groupby(labels).mean().add_prefix("mean_")
        profile = profile.join(means, on="regime")
    return profile


def agreement(labels: pd.DataFrame) -> pd.DataFrame:
    """Adjusted Rand index between every two labelings (columns), on days all of them label.
    1 means the same partition up to renaming, 0 what chance would give."""
    labels = labels.dropna()
    names = list(labels.columns)
    return pd.DataFrame(
        [[adjusted_rand_score(labels[a], labels[b]) for b in names] for a in names],
        index=names, columns=names,
    )
