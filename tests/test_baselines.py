import numpy as np
import pandas as pd
import pytest

from fxml.models import baselines as bl
from fxml.validation.walkforward import walk_forward

TEST_START = "2011-09-01"


def folds_for(df):
    return list(walk_forward(df.index, df["target_date"], TEST_START))


@pytest.fixture(scope="module")
def eurusd(synthetic_table):
    return synthetic_table.xs("EURUSD", level="pair")


def fast_arima(df, folds):
    return bl.ret_arima(df, folds, max_lag=1)


MODELS = [bl.vol_naive, bl.vol_ewma, bl.vol_garch, bl.vol_gjr_garch, bl.vol_har,
          bl.ret_zero, bl.ret_mean, fast_arima, bl.dir_frequency, bl.dir_persistence]


@pytest.mark.parametrize("model", MODELS)
def test_one_forecast_per_test_day(eurusd, model):
    folds = folds_for(eurusd)
    forecast = model(eurusd, folds)
    expected = eurusd.index[np.concatenate([f.test for f in folds])]
    assert forecast.index.equals(expected)
    assert forecast.notna().all()


@pytest.mark.parametrize("model", MODELS)
def test_forecasts_never_use_the_future(eurusd, model):
    """Scrambling data after the first test month must not change that month's forecasts."""
    folds = folds_for(eurusd)
    first = folds[0]
    end = eurusd.index[first.test[-1]]
    scrambled = eurusd.copy()
    later = scrambled.index > end
    rng = np.random.default_rng(3)
    for col in ["ret", "rv", "logrv_d", "logrv_w", "logrv_m", "y_rv", "y_logrv", "y_ret"]:
        scrambled.loc[later, col] = scrambled.loc[later, col] * rng.uniform(0.5, 1.5, later.sum())
    original = model(eurusd, [first])
    rerun = model(scrambled, [first])
    pd.testing.assert_series_equal(original, rerun)


def test_volatility_forecasts_are_positive_variances(eurusd):
    folds = folds_for(eurusd)
    for model in (bl.vol_ewma, bl.vol_garch, bl.vol_gjr_garch, bl.vol_har):
        forecast = model(eurusd, folds)
        assert (forecast > 0).all()
        assert forecast.median() == pytest.approx(eurusd["rv"].median(), rel=1.0)


def test_har_recovers_persistence(eurusd):
    """On data with volatility clustering, HAR should beat the naive forecast on QLIKE."""
    from fxml.validation.metrics import qlike_losses

    folds = folds_for(eurusd)
    actual = eurusd["y_rv"].iloc[np.concatenate([f.test for f in folds])]
    har = qlike_losses(actual, bl.vol_har(eurusd, folds)).mean()
    naive = qlike_losses(actual, bl.vol_naive(eurusd, folds)).mean()
    assert har < naive
