"""Clustering models of the market state, with labels ordered from calm to turbulent.

Four methods are compared:
- k-means: hard assignments to the nearest centre;
- Ward: agglomerative hierarchical clustering; new days go to the nearest cluster centre;
- Gaussian mixture fitted by EM: soft assignments with a probability for each regime;
- DBSCAN: density-based. Days far from any dense region are noise (-1), which flags unusual,
  shock-like days. New days join the cluster of the nearest core day within ``eps``, else noise.

Every model standardises the state on its training window. Regimes are numbered by the
training days' average short-horizon volatility, so regime 0 is always the calmest whatever
the clustering's internal numbering, and labels stay comparable across refits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

METHODS = ("kmeans", "ward", "gmm", "dbscan")
ORDER_BY = "vol_short"


def knee_distance(z: np.ndarray, min_samples: int) -> float:
    """DBSCAN radius from the knee of the sorted distances to each point's
    ``min_samples``-th neighbour: the point furthest from the line joining the curve's ends."""
    distances = NearestNeighbors(n_neighbors=min_samples).fit(z).kneighbors(z)[0][:, -1]
    curve = np.sort(distances)
    x = np.linspace(0, 1, len(curve))
    y = (curve - curve[0]) / (curve[-1] - curve[0])
    return float(curve[np.argmax(x - y)])


@dataclass
class RegimeModel:
    method: str
    n_regimes: int | None = None  # not used by DBSCAN
    seed: int = 42
    min_samples: int = 10  # DBSCAN only
    params_: dict = field(default_factory=dict, init=False)

    def fit(self, state: pd.DataFrame) -> RegimeModel:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method}")
        self.columns_ = list(state.columns)
        self.scaler_ = StandardScaler().fit(state)
        z = self.scaler_.transform(state)
        k = self.n_regimes
        if self.method == "kmeans":
            self.model_ = KMeans(n_clusters=k, n_init=20, random_state=self.seed).fit(z)
            raw = self.model_.labels_
        elif self.method == "ward":
            raw = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(z)
            self.centres_ = np.vstack([z[raw == c].mean(axis=0) for c in range(k)])
        elif self.method == "gmm":
            self.model_ = GaussianMixture(
                n_components=k, covariance_type="full", n_init=5, random_state=self.seed
            ).fit(z)
            raw = self.model_.predict(z)
        else:
            eps = knee_distance(z, self.min_samples)
            fitted = DBSCAN(eps=eps, min_samples=self.min_samples).fit(z)
            raw = fitted.labels_
            self.core_ = z[fitted.core_sample_indices_]
            self.core_labels_ = raw[fitted.core_sample_indices_]
            self.params_ = {"eps": eps}
        self._set_order(state[ORDER_BY].to_numpy(), raw)
        self.labels_ = self._relabel(raw)
        return self

    def _set_order(self, volatility: np.ndarray, raw: np.ndarray) -> None:
        clusters = np.unique(raw[raw >= 0])
        means = [volatility[raw == c].mean() for c in clusters]
        self.perm_ = clusters[np.argsort(means)]  # new label i is old cluster perm_[i]
        self.mapping_ = {old: new for new, old in enumerate(self.perm_)} | {-1: -1}

    def _relabel(self, raw: np.ndarray) -> np.ndarray:
        return np.array([self.mapping_[c] for c in raw], dtype=int)

    def predict(self, state: pd.DataFrame) -> np.ndarray:
        z = self.scaler_.transform(state[self.columns_])
        if self.method in ("kmeans", "gmm"):
            raw = self.model_.predict(z)
        elif self.method == "ward":
            raw = np.argmin(((z[:, None, :] - self.centres_[None]) ** 2).sum(axis=2), axis=1)
        else:
            distance, index = NearestNeighbors(n_neighbors=1).fit(self.core_).kneighbors(z)
            near = distance[:, 0] <= self.params_["eps"]
            raw = np.where(near, self.core_labels_[index[:, 0]], -1)
        return self._relabel(raw)

    def predict_proba(self, state: pd.DataFrame) -> np.ndarray:
        """Regime probabilities, columns ordered calm to turbulent (Gaussian mixture only)."""
        if self.method != "gmm":
            raise ValueError("only the Gaussian mixture gives probabilities")
        z = self.scaler_.transform(state[self.columns_])
        return self.model_.predict_proba(z)[:, self.perm_]


def walk_forward_labels(
    state: pd.DataFrame, method: str, n_regimes: int | None, first_refit, freq: str = "YS",
    seed: int = 42,
) -> pd.DataFrame:
    """Regime labels where every day is labelled by a model fitted only on earlier days.

    The model is refitted at each period start from ``first_refit`` on all complete days before
    it. Days before ``first_refit`` get the first model's own (in-sample) labels and are flagged.
    Gaussian mixture output adds one probability column per regime.
    """
    state = state.dropna()
    starts = pd.date_range(pd.Timestamp(first_refit), state.index.max(), freq=freq)
    frames = []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else pd.Timestamp.max
        model = RegimeModel(method, n_regimes, seed=seed).fit(state[state.index < start])
        period = state[(state.index >= start) & (state.index < stop)]
        blocks = [(period, False)]
        if i == 0:
            blocks.insert(0, (state[state.index < start], True))
        for block, in_sample in blocks:
            out = pd.DataFrame({"regime": model.predict(block), "in_sample": in_sample},
                               index=block.index)
            if method == "gmm":
                proba = model.predict_proba(block)
                for r in range(proba.shape[1]):
                    out[f"p{r}"] = proba[:, r]
            frames.append(out)
    return pd.concat(frames)
