"""Classical machine learning models for the three tasks, run through the walk-forward folds.

For each pair and model:

- hyperparameters are tuned once a year, at the year's first fold: every candidate is fitted
  on the fold's fitting rows and scored on its validation slice, and the best one is used for
  every refit that year;
- the model is refitted on all training rows of each fold (``refit_months`` = 1) and forecasts
  that fold's test month.

Volatility models predict the next day's log realized variance. Turning that into a variance
needs a scale factor, because exp(E[log RV]) is below E[RV]. The factor makes the validation
forecasts unbiased on average (the mean ratio of realized to forecast variance, which is also
the scale that minimises QLIKE). It is refitted with the hyperparameters each year.

Regime-aware variants come in two forms:

- ``regime="feature"``: the global model plus the regime probabilities as features;
- ``regime="experts"``: one model per regime, trained on that regime's days. Forecasts are
  averaged with the day's regime probabilities as weights. A regime with too few training days
  falls back to a model trained on all days.
"""

from __future__ import annotations

import itertools
import json
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from fxml.models.inputs import BASE_FEATURES, regime_columns, regime_features
from fxml.validation.metrics import qlike_losses
from fxml.validation.walkforward import Fold

TARGETS = {"volatility": "y_logrv", "direction": "y_dir", "returns": "y_ret"}
# Early tuning windows end before the S&P 500 data starts (November 2010); the imputer drops
# those empty columns for that fit, which is the intended behaviour.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
MIN_EXPERT_ROWS = 250  # about a year of days; fewer and the regime uses the all-days model
EPS = 1e-6


@dataclass(frozen=True)
class Spec:
    name: str
    task: str
    make: Callable[[dict, int], object]  # (hyperparameters, seed) -> unfitted estimator
    grid: tuple[dict, ...] = ({},)
    regime: str | None = None  # None, "feature" or "experts"
    refit_months: int = 1


def grid(**options) -> tuple[dict, ...]:
    keys = list(options)
    return tuple(dict(zip(keys, values, strict=True))
                 for values in itertools.product(*options.values()))


def scaled(estimator):
    """Median imputation and standardisation, both fitted on the training rows only."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def imputed(estimator):
    return make_pipeline(SimpleImputer(strategy="median"), estimator)


def lightgbm(task: str, params: dict, seed: int):
    import lightgbm as lgb  # imported here: it needs the OpenMP runtime, missing on some Macs

    model = lgb.LGBMClassifier if task == "direction" else lgb.LGBMRegressor
    return model(learning_rate=0.03, min_child_samples=40, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=1, verbose=-1,
                 **params)


def calibrated(estimator):
    """Platt-scaled probabilities for a classifier that only gives scores (the SVMs)."""
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=3, ensemble=False)


LIGHTGBM_GRID = grid(num_leaves=[7, 15], n_estimators=[150, 400])
FOREST_TREES = 300


def _ridge(p, s):
    return scaled(Ridge(**p))


def _lgb_volatility(p, s):
    return lightgbm("volatility", p, s)


def _logistic(p, s):
    return scaled(LogisticRegression(max_iter=2000, **p))


def _lgb_direction(p, s):
    return lightgbm("direction", p, s)


def _lgb_returns(p, s):
    return lightgbm("returns", p, s)


def _with_regimes(spec: Spec) -> list[Spec]:
    """The global spec plus its two regime-aware variants."""
    return [
        spec,
        Spec(f"{spec.name}_regime", spec.task, spec.make, spec.grid, "feature",
             spec.refit_months),
        Spec(f"{spec.name}_experts", spec.task, spec.make, spec.grid, "experts",
             spec.refit_months),
    ]


SPECS: list[Spec] = [
    # volatility: next day log realized variance
    *_with_regimes(Spec("ridge", "volatility", _ridge, grid(alpha=[1, 10, 100, 1000]))),
    Spec("lasso", "volatility", lambda p, s: scaled(Lasso(max_iter=10000, **p)),
         grid(alpha=[0.001, 0.003, 0.01, 0.03])),
    Spec("knn", "volatility", lambda p, s: scaled(KNeighborsRegressor(**p)),
         grid(n_neighbors=[20, 50, 100])),
    Spec("random_forest", "volatility",
         lambda p, s: imputed(RandomForestRegressor(n_estimators=FOREST_TREES, max_features=0.33,
                                                    n_jobs=1, random_state=s, **p)),
         grid(min_samples_leaf=[10, 30, 100])),
    *_with_regimes(Spec("lightgbm", "volatility", _lgb_volatility, LIGHTGBM_GRID)),
    # direction: probability that the next day's return is positive
    *_with_regimes(Spec("logistic_l2", "direction", _logistic, grid(C=[0.001, 0.01, 0.1, 1]))),
    Spec("logistic_l1", "direction",
         lambda p, s: scaled(LogisticRegression(l1_ratio=1.0, solver="liblinear", **p)),
         grid(C=[0.003, 0.01, 0.03, 0.1])),
    Spec("naive_bayes", "direction", lambda p, s: imputed(GaussianNB(**p)),
         grid(var_smoothing=[1e-9, 1e-6, 1e-3])),
    Spec("knn", "direction", lambda p, s: scaled(KNeighborsClassifier(**p)),
         grid(n_neighbors=[50, 100, 200])),
    Spec("tree", "direction",
         lambda p, s: imputed(DecisionTreeClassifier(min_samples_leaf=50, random_state=s, **p)),
         grid(max_depth=[2, 3, 5])),
    Spec("svm_linear", "direction", lambda p, s: scaled(calibrated(LinearSVC(**p))),
         grid(C=[0.001, 0.01, 0.1])),
    Spec("svm_rbf", "direction",
         lambda p, s: scaled(calibrated(SVC(kernel="rbf", cache_size=500, random_state=s, **p))),
         grid(C=[0.1, 1.0])),
    Spec("random_forest", "direction",
         lambda p, s: imputed(RandomForestClassifier(n_estimators=FOREST_TREES,
                                                     max_features="sqrt", n_jobs=1,
                                                     random_state=s, **p)),
         grid(min_samples_leaf=[20, 50, 100])),
    *_with_regimes(Spec("lightgbm", "direction", _lgb_direction, LIGHTGBM_GRID)),
    # returns: the next day's log return
    *_with_regimes(Spec("ridge", "returns", _ridge, grid(alpha=[1e2, 1e3, 1e4, 1e5]))),
    *_with_regimes(Spec("lightgbm", "returns", _lgb_returns, LIGHTGBM_GRID)),
]


def get_spec(task: str, name: str) -> Spec:
    return next(s for s in SPECS if s.task == task and s.name == name)


# ---------------------------------------------------------------- fitting and forecasting


def feature_columns(spec: Spec, frame: pd.DataFrame) -> list[str]:
    return BASE_FEATURES + (regime_features(frame) if spec.regime == "feature" else [])


def _matrix(rows: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return rows[columns].to_numpy(dtype=float)


def _fit_one(spec: Spec, params: dict, rows: pd.DataFrame, columns: list[str], seed: int):
    target = rows[TARGETS[spec.task]]
    return spec.make(params, seed).fit(_matrix(rows, columns), target.to_numpy())


def fit(spec: Spec, params: dict, rows: pd.DataFrame, seed: int = 42) -> dict:
    """Fitted model(s): ``{None: model}``, or for experts ``{regime: model, "all": model}``."""
    rows = rows[rows[TARGETS[spec.task]].notna()]
    columns = feature_columns(spec, rows)
    if spec.regime != "experts":
        return {None: _fit_one(spec, params, rows, columns, seed)}
    models = {}
    for regime in range(len(regime_columns(rows))):
        subset = rows[rows["regime"] == regime]
        enough = len(subset) >= MIN_EXPERT_ROWS
        if spec.task == "direction":
            enough &= subset[TARGETS[spec.task]].nunique() == 2
        models[regime] = _fit_one(spec, params, subset, columns, seed) if enough else None
    if any(model is None for model in models.values()):
        models["all"] = _fit_one(spec, params, rows, columns, seed)
    return models


def _predict_one(task: str, model, x: np.ndarray) -> np.ndarray:
    if task == "direction":
        up = list(model.classes_).index(1)
        return model.predict_proba(x)[:, up]
    return model.predict(x)


def predict(spec: Spec, models: dict, rows: pd.DataFrame) -> np.ndarray:
    """Raw predictions: log variance, probability of an up day, or return."""
    x = _matrix(rows, feature_columns(spec, rows))
    if spec.regime != "experts":
        return _predict_one(spec.task, models[None], x)
    probs = rows[regime_columns(rows)].to_numpy(dtype=float, copy=True)
    unknown = np.isnan(probs).any(axis=1)
    probs[unknown] = 1 / probs.shape[1]  # no regime information: weight the experts equally
    out = np.zeros(len(rows))
    for regime in range(probs.shape[1]):
        model = models[regime] if models[regime] is not None else models["all"]
        out += probs[:, regime] * _predict_one(spec.task, model, x)
    return out


def to_forecast(task: str, raw: np.ndarray, scale: float) -> np.ndarray:
    """Forecast in the units the baselines use: variance, probability or log return."""
    if task == "volatility":
        return np.exp(raw) * scale
    if task == "direction":
        return np.clip(raw, 0.0, 1.0)
    return raw


def validation_loss(task: str, raw: np.ndarray, rows: pd.DataFrame) -> tuple[float, float]:
    """(loss, scale) on the validation slice: QLIKE after scaling for volatility, log loss for
    direction and squared error for returns."""
    if task == "volatility":
        realized, base = rows["y_rv"].to_numpy(), np.exp(raw)
        scale = float(np.mean(realized / base))
        return float(qlike_losses(realized, base * scale).mean()), scale
    actual = rows[TARGETS[task]].to_numpy()
    if task == "direction":
        return float(log_loss(actual, np.clip(raw, EPS, 1 - EPS), labels=[0, 1])), 1.0
    return float(np.mean((actual - raw) ** 2)), 1.0


def tune(spec: Spec, frame: pd.DataFrame, fold: Fold, seed: int = 42) -> tuple[dict, float, float]:
    """Best hyperparameters on the fold's validation slice: (params, scale, validation loss)."""
    if len(spec.grid) == 1 and spec.task != "volatility":
        return spec.grid[0], 1.0, np.nan
    fitting = frame.iloc[fold.fit]
    valid = frame.iloc[fold.valid]
    valid = valid[valid[TARGETS[spec.task]].notna()]
    best = None
    for params in spec.grid:
        raw = predict(spec, fit(spec, params, fitting, seed), valid)
        loss, scale = validation_loss(spec.task, raw, valid)
        if best is None or loss < best[2]:
            best = (params, scale, loss)
    return best


def run_spec(
    spec: Spec, frame: pd.DataFrame, folds: Sequence[Fold], seed: int = 42
) -> tuple[pd.Series, pd.DataFrame]:
    """Forecasts for every test row of ``folds``, and a log of each year's tuning."""
    forecasts, log = [], []
    year, models, age, params, scale = None, None, 0, None, 1.0
    for fold in folds:
        if fold.test_start.year != year:
            year = fold.test_start.year
            params, scale, loss = tune(spec, frame, fold, seed)
            log.append({"year": year, "params": json.dumps(params, sort_keys=True),
                        "scale": scale, "validation_loss": loss})
            models = None
        if models is None or age >= spec.refit_months:
            models, age = fit(spec, params, frame.iloc[fold.train], seed), 0
        age += 1
        test = frame.iloc[fold.test]
        raw = predict(spec, models, test)
        forecasts.append(pd.Series(to_forecast(spec.task, raw, scale), index=test.index))
    return pd.concat(forecasts), pd.DataFrame(log)
