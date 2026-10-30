import numpy as np
import pandas as pd
import pytest

from fxml.factors import CurrencyFactors, average_correlation, usd_value_returns

from .conftest import PAIRS


def common_factor_returns(n=1500, seed=0):
    """Five currencies driven by a shared "dollar" factor plus a second factor that splits them."""
    rng = np.random.default_rng(seed)
    dollar, split = rng.normal(0, 1, n), rng.normal(0, 1, n)
    beta = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    tilt = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    noise = rng.normal(0, 0.4, (n, 5))
    values = (dollar[:, None] * beta + 0.6 * split[:, None] * tilt + noise) * 0.005
    dates = pd.bdate_range("2010-01-01", periods=n, name="date")
    return pd.DataFrame(values, index=dates, columns=["EUR", "GBP", "JPY", "AUD", "NZD"])


def test_dollar_based_pairs_change_sign(synthetic_table):
    out = usd_value_returns(synthetic_table, PAIRS)
    assert list(out.columns) == ["EUR", "JPY"]
    wide = synthetic_table["ret"].unstack("pair")
    pd.testing.assert_series_equal(out["EUR"], wide["EURUSD"], check_names=False)
    pd.testing.assert_series_equal(out["JPY"], -wide["USDJPY"], check_names=False)


def test_first_component_is_the_shared_factor():
    returns = common_factor_returns()
    factors = CurrencyFactors(n_components=3).fit(returns)
    assert factors.explained_variance_ratio_["pc1"] > 0.5
    assert (factors.loadings_["pc1"] > 0).all()  # everyone rises against the dollar together
    assert factors.explained_variance_ratio_.is_monotonic_decreasing


def test_signs_do_not_depend_on_the_data_orientation():
    """Flipping every return, or reordering the columns, gives the same loadings."""
    returns = common_factor_returns()
    base = CurrencyFactors(3).fit(returns).loadings_
    flipped = CurrencyFactors(3).fit(-returns).loadings_
    shuffled = CurrencyFactors(3).fit(returns[returns.columns[::-1]]).loadings_
    pd.testing.assert_frame_equal(base, flipped)
    pd.testing.assert_frame_equal(base, shuffled.loc[base.index])
    for pc in ["pc2", "pc3"]:
        column = base[pc]
        assert column[column.abs().idxmax()] > 0


def test_transform_uses_only_each_days_returns():
    returns = common_factor_returns()
    factors = CurrencyFactors(3).fit(returns.iloc[:1000])
    full = factors.transform(returns)
    one_day = factors.transform(returns.iloc[[1200]])
    pd.testing.assert_frame_equal(full.iloc[[1200]], one_day)


def test_average_correlation():
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    same = pd.DataFrame({"a": x, "b": 2 * x, "c": x + 1})
    out = average_correlation(same, window=20)
    assert out.iloc[:19].isna().all()
    assert out.iloc[19:].to_numpy() == pytest.approx(1.0)
    independent = pd.DataFrame(rng.normal(size=(5000, 4)), columns=list("abcd"))
    assert abs(average_correlation(independent, window=1000).dropna().mean()) < 0.03
