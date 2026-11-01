"""Hidden Markov model regimes, fitted with the Baum-Welch EM algorithm.

A Gaussian mixture treats every day on its own. A hidden Markov model also learns how regimes
follow one another: a transition matrix gives the chance of moving from each regime to each
other one the next day, so persistence is part of the model rather than something found
afterwards.

Forecasting must not use the future, so the regime probabilities here are *filtered*: the
probability of each regime on day t given the market state up to day t only (the forward
algorithm). The smoothed probabilities that hmmlearn's ``predict_proba`` returns condition on
the whole sample, later days included, and are never used.

Like the other regime models, the state is standardised on the training window and regimes are
numbered by their average short horizon volatility, so regime 0 is the calmest.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from fxml.regimes.models import ORDER_BY
from fxml.validation.walkforward import refit_periods

# hmmlearn 0.3.3 reshapes arrays in a way NumPy 2.5 deprecates; the results are unaffected.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hmmlearn")


class HMMRegimes:
    def __init__(self, n_regimes: int, seed: int = 42, restarts: int = 5, iterations: int = 200):
        self.n_regimes = n_regimes
        self.seed = seed
        self.restarts = restarts
        self.iterations = iterations

    def fit(self, state: pd.DataFrame) -> HMMRegimes:
        """Baum-Welch from several random starts, keeping the highest likelihood."""
        self.columns_ = list(state.columns)
        self.scaler_ = StandardScaler().fit(state)
        z = self.scaler_.transform(state)
        best, best_score = None, -np.inf
        for restart in range(self.restarts):
            model = GaussianHMM(n_components=self.n_regimes, covariance_type="full",
                                n_iter=self.iterations, tol=1e-4,
                                random_state=self.seed + restart).fit(z)
            score = model.score(z)
            if score > best_score:
                best, best_score = model, score
        self.model_, self.log_likelihood_ = best, best_score
        self.order_ = np.argsort(best.means_[:, self.columns_.index(ORDER_BY)])
        return self

    def bic(self, state: pd.DataFrame) -> float:
        """Bayesian information criterion on ``state`` (lower is better)."""
        k, d = self.n_regimes, len(self.columns_)
        parameters = (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) / 2
        log_likelihood = self.model_.score(self.scaler_.transform(state[self.columns_]))
        return float(-2 * log_likelihood + parameters * np.log(len(state)))

    def transition_matrix(self) -> pd.DataFrame:
        """Estimated chance of moving from each regime (rows) to each (columns) the next day."""
        matrix = self.model_.transmat_[np.ix_(self.order_, self.order_)]
        names = range(self.n_regimes)
        return pd.DataFrame(matrix, index=pd.Index(names, name="from"),
                            columns=pd.Index(names, name="to"))

    def filter(self, state: pd.DataFrame) -> np.ndarray:
        """Filtered regime probabilities, one row per day, columns calm to turbulent.

        Each row uses only that day and the days before it in ``state``."""
        z = self.scaler_.transform(state[self.columns_])
        model = self.model_
        log_emission = np.column_stack([
            multivariate_normal(model.means_[j], model.covars_[j]).logpdf(z)
            for j in range(self.n_regimes)
        ]).reshape(len(z), self.n_regimes)
        with np.errstate(divide="ignore"):  # an impossible start or move has log probability -inf
            log_start, log_transition = np.log(model.startprob_), np.log(model.transmat_)
        out = np.empty_like(log_emission)
        log_alpha = log_start + log_emission[0]
        out[0] = log_alpha - logsumexp(log_alpha)
        for t in range(1, len(z)):
            predicted = logsumexp(out[t - 1][:, None] + log_transition, axis=0)
            log_alpha = predicted + log_emission[t]
            out[t] = log_alpha - logsumexp(log_alpha)
        return np.exp(out)[:, self.order_]


def walk_forward_hmm(
    state: pd.DataFrame, n_regimes: int, first_refit, freq: str = "YS", seed: int = 42
) -> pd.DataFrame:
    """Filtered regime probabilities where every day uses a model fitted only on earlier days.

    Refits at ``first_refit`` and each ``freq`` boundary. The filter runs from the start of the
    data each time, so its probabilities settle before the period they are used for. Days
    before ``first_refit`` get the first model's (in-sample) probabilities and are flagged.
    """
    state = state.dropna()
    frames = []
    for i, (start, stop) in enumerate(refit_periods(first_refit, state.index.max(), freq)):
        model = HMMRegimes(n_regimes, seed).fit(state[state.index < start])
        through = state[state.index < stop]
        probs = model.filter(through)
        keep = through.index < start if i == 0 else through.index >= start
        blocks = [(keep, i == 0)]
        if i == 0:
            blocks.append((through.index >= start, False))
        for mask, in_sample in blocks:
            out = pd.DataFrame(probs[mask], index=through.index[mask],
                               columns=[f"hmm_p{r}" for r in range(n_regimes)])
            out.insert(0, "in_sample", in_sample)
            out.insert(0, "regime", probs[mask].argmax(axis=1))
            frames.append(out)
    return pd.concat(frames).sort_index()
