from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fxml.models import baselines, horizons

from .conftest import CUT

WF = SimpleNamespace(test_start=pd.Timestamp("2012-01-01"), refit_frequency="MS",
                     validation_fraction=0.2, embargo_days=5)


def test_horizon_targets(frame):
    data = horizons.horizon_frame(frame, 5)
    day = frame.index[100]
    ahead = frame.iloc[101:106]
    assert data.loc[day, "y_rv"] == pytest.approx(ahead["rv"].mean())
    assert data.loc[day, "y_ret"] == pytest.approx(ahead["ret"].sum())
    assert data.loc[day, "y_dir"] == float(ahead["ret"].sum() > 0)
    assert data.loc[day, "target_date"] == frame.index[105]
    assert data["y_rv"].iloc[-5:].isna().all()  # the last days have no complete target


@pytest.mark.parametrize("h", [5, 22])
def test_folds_purge_and_embargo_by_the_horizon(frame, h):
    data = horizons.horizon_frame(frame, h)
    folds = horizons.horizon_folds(data, h, WF)
    for fold in folds:
        start = fold.test_start
        assert (data["target_date"].iloc[fold.train] < start).all()  # targets known in time
        last_fit_target = data["target_date"].iloc[fold.fit].max()
        assert last_fit_target < data.index[fold.valid[0]]  # fitting targets end before validation


def test_garch_horizon_averages_the_step_forecasts(frame, folds):
    one = baselines.vol_garch(frame, folds[:1])
    same = baselines._garch(frame, folds[:1], asymmetric=False, horizon=1)
    pd.testing.assert_series_equal(one, same)
    week = baselines.vol_garch(frame, folds[:1], horizon=5)
    assert (week > 0).all()
    assert not np.allclose(week, one)  # later steps revert towards the long-run variance


def lightgbm_loads() -> bool:
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):  # unloadable without the OpenMP runtime on some Macs
        return False
    return True


needs_lightgbm = pytest.mark.skipif(not lightgbm_loads(), reason="LightGBM cannot load here")


@needs_lightgbm
def test_every_horizon_model_forecasts_every_test_day(frame):
    out = horizons.run_horizon(frame.loc[:"2012-03-31"], 5, WF, seed=0)
    for task, models in [("volatility", horizons.VOLATILITY_MODELS),
                         ("direction", horizons.DIRECTION_MODELS)]:
        assert set(out.loc[out["task"] == task, "model"]) == set(models)
    assert out.groupby(["task", "model"]).size().nunique() == 1  # same test days for all
    assert np.isfinite(out["forecast"]).all()
    assert (out.loc[out["task"] == "volatility", "forecast"] > 0).all()


@needs_lightgbm
def test_horizon_forecasts_never_use_the_future(frame):
    small = frame.loc[:"2012-03-31"]
    before = horizons.run_horizon(small, 5, WF, seed=0)
    rng = np.random.default_rng(0)
    scrambled = small.copy()
    later = scrambled.index > CUT
    for column in ["rv", "ret", "logrv_d", "logrv_w", "logrv_m"]:
        scrambled.loc[later, column] *= rng.uniform(0.5, 1.5, later.sum())
    after = horizons.run_horizon(scrambled, 5, WF, seed=0)

    def upto_cut(d):
        return d[d["date"] <= CUT].set_index(["task", "model", "date"])["forecast"].sort_index()

    pd.testing.assert_series_equal(upto_cut(before), upto_cut(after))
