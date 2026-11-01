import numpy as np
import pandas as pd
import pytest

from fxml.features.fitted import walk_forward_garch
from fxml.models import ml
from fxml.models.inputs import BASE_FEATURES, FACTOR_COLUMNS
from fxml.models.learning_curves import (
    curve_targets,
    learning_curve_forecasts,
    summarise_curves,
)

from .conftest import CUT, FIRST_REFIT, HAS_LIGHTGBM


def test_garch_feature_is_causal_and_finite(frame):
    ret = frame["ret"]
    full = walk_forward_garch(ret, FIRST_REFIT)
    short = walk_forward_garch(ret[ret.index <= CUT], FIRST_REFIT)
    pd.testing.assert_series_equal(full[full.index <= CUT], short)
    assert np.isfinite(full.iloc[1:]).all()
    # the variance forecast tracks realized variance
    assert np.corrcoef(full.iloc[1:], np.log(frame["rv"].iloc[1:]))[0, 1] > 0.3


def test_pair_frame_carries_regimes_forward_but_not_factor_returns(frame, state):
    missing = frame.index.difference(state.index)
    assert len(missing) >= 3
    day = missing[0]
    previous = state.index[state.index < day][-1]
    assert frame.loc[day, "gmm_p1"] == state.loc[previous, "gmm_p1"]
    assert frame.loc[day, FACTOR_COLUMNS].isna().all()
    assert frame.loc[day, "regime"] == int(state.loc[previous, "gmm_p1"] > 0.5)
    assert set(BASE_FEATURES) <= set(frame.columns)


SPEC_IDS = [f"{s.task}/{s.name}" for s in ml.SPECS]


@pytest.mark.parametrize("spec", ml.SPECS, ids=SPEC_IDS)
def test_every_model_forecasts_every_test_day(spec, frame, folds):
    if "lightgbm" in spec.name and not HAS_LIGHTGBM:
        pytest.skip("LightGBM cannot load here")
    few = folds[:2]
    forecast, log = ml.run_spec(spec, frame, few)
    expected = np.concatenate([fold.test for fold in few])
    assert forecast.index.equals(frame.index[expected])
    assert np.isfinite(forecast).all()
    if spec.task == "volatility":
        assert (forecast > 0).all()
    if spec.task == "direction":
        assert ((forecast >= 0) & (forecast <= 1)).all()
    assert len(log) == 1 and log["scale"].iloc[0] > 0


@pytest.mark.parametrize("task, name", [
    ("volatility", "ridge"), ("direction", "logistic_l2_experts"), ("returns", "ridge_regime"),
])
def test_forecasts_never_use_the_future(task, name, frame, folds):
    """Scrambling every day after CUT leaves forecasts up to CUT unchanged."""
    spec = ml.get_spec(task, name)
    some = [f for f in folds if f.test_start <= CUT]
    before, _ = ml.run_spec(spec, frame, some)
    rng = np.random.default_rng(0)
    scrambled = frame.copy()
    later = scrambled.index > CUT
    columns = [*BASE_FEATURES, "gmm_p0", "gmm_p1", "y_rv", "y_logrv", "y_ret"]
    scrambled.loc[later, columns] *= rng.uniform(0.5, 1.5, (later.sum(), len(columns)))
    scrambled.loc[later, "y_dir"] = 1 - scrambled.loc[later, "y_dir"]
    after, _ = ml.run_spec(spec, scrambled, some)
    keep = before.index <= CUT
    pd.testing.assert_series_equal(before[keep], after[keep])


def test_volatility_scale_makes_validation_forecasts_unbiased(frame, folds):
    spec = ml.get_spec("volatility", "ridge")
    params, scale, loss = ml.tune(spec, frame, folds[0])
    assert params in spec.grid
    valid = frame.iloc[folds[0].valid]
    raw = ml.predict(spec, ml.fit(spec, params, frame.iloc[folds[0].fit]), valid)
    assert np.mean(valid["y_rv"] / (np.exp(raw) * scale)) == pytest.approx(1.0)
    assert 0 < loss < 1


def test_experts_weight_by_regime_and_fall_back(frame, folds, monkeypatch):
    spec = ml.get_spec("volatility", "ridge_experts")
    train = frame.iloc[folds[0].train]
    models = ml.fit(spec, {"alpha": 10}, train)
    assert set(models) >= {0, 1}
    test = frame.iloc[folds[0].test].copy()
    test[["gmm_p0", "gmm_p1"]] = [1.0, 0.0]
    columns = ml.feature_columns(spec, test)
    calm_only = models[0].predict(test[columns].to_numpy(dtype=float))
    np.testing.assert_allclose(ml.predict(spec, models, test), calm_only)
    # with too few days in a regime, that regime uses the model trained on all days
    monkeypatch.setattr(ml, "MIN_EXPERT_ROWS", 10**6)
    sparse = ml.fit(spec, {"alpha": 10}, train)
    assert sparse[0] is None and sparse[1] is None and "all" in sparse
    everyone = sparse["all"].predict(test[columns].to_numpy(dtype=float))
    np.testing.assert_allclose(ml.predict(spec, sparse, test), everyone)


def test_learning_curves_share_test_days(frame, folds):
    curves = learning_curve_forecasts(frame, folds[:2], sizes=(21, 63, 252))
    assert set(curves["size"]) == {21, 63, 252, 0}
    assert set(curves["target"]) == {"direction", "high_volatility"}
    days = curves.groupby(["target", "model", "size"])["date"].apply(lambda d: tuple(sorted(d)))
    assert days.groupby(level="target").nunique().eq(1).all()
    small = curves[curves["size"] == 21]
    assert (small["train_days"] == 21).all()
    summary = summarise_curves(curves)
    assert len(summary) == 16
    assert summary["accuracy"].between(0, 1).all()
    # volatility clusters, so the high volatility target is learnable from the full window
    full = summary[(summary["target"] == "high_volatility") & (summary["size"] == 0)]
    assert (full["roc_auc"] > 0.6).all()


def test_high_volatility_target_uses_a_threshold_known_at_the_close(frame):
    targets = curve_targets(frame)
    threshold = frame["logrv_d"].rolling(63).median()
    day = frame.index[200]
    assert targets.loc[day, "high_volatility"] == float(frame.loc[day, "y_logrv"] > threshold[day])
    assert targets["high_volatility"].iloc[:62].isna().all()
