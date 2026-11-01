import importlib.util

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fxml.backtest import performance as perf
from fxml.backtest import risk
from fxml.backtest import strategies as st
from fxml.validation.stats_tests import christoffersen

try:
    import lightgbm  # noqa: F401

    HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
except OSError:  # installed but unloadable (no OpenMP runtime on some Macs)
    HAS_LIGHTGBM = False


def toy_inputs(n=600, seed=0):
    """A pair's modelling frame, closes and forecasts in the shapes run_backtest.py reads."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n, name="date")
    variance = (0.006 * np.exp(rng.normal(0, 0.3, n))) ** 2
    ret = rng.normal(0, np.sqrt(variance))
    frame = pd.DataFrame({
        "y_ret": ret,
        "rate_diff": np.where(np.arange(n) < n // 2, 1.5, -0.5),
        "target_date": dates.to_series().shift(-1).fillna(dates[-1] + pd.offsets.BDay()),
        "mom_252": rng.normal(0, 0.05, n),
        "regime": rng.integers(0, 2, n).astype(float),
    }, index=dates)
    close = pd.Series(1.2, index=dates)
    rows = []
    for model in st.DIRECTION_MODELS:
        rows.append(pd.DataFrame({"task": "direction", "model": model, "date": dates,
                                  "forecast": 0.5 + rng.normal(0, 0.02, n)}))
    for model in st.SIZING_MODELS.values():
        rows.append(pd.DataFrame({"task": "volatility", "model": model, "date": dates,
                                  "forecast": variance * rng.lognormal(0, 0.1, n)}))
    return frame, close, pd.concat(rows, ignore_index=True), variance


@pytest.fixture(scope="module")
def data():
    frame, close, predictions, _ = toy_inputs()
    return st.pair_data(frame, close, predictions, spread=0.0002)


def test_pair_data_carry_and_costs(data):
    friday, monday = pd.Timestamp("2012-01-06"), pd.Timestamp("2012-01-09")
    assert data.loc[friday, "carry"] == pytest.approx(1.5 / 100 * 3 / 365)  # weekend accrues
    assert data.loc[monday, "carry"] == pytest.approx(1.5 / 100 / 365)
    assert data["half_spread"].iloc[0] == pytest.approx(0.0001 / 1.2)
    assert {"p_lightgbm", "var_baseline_har", "var_lightgbm"} <= set(data.columns)


def test_signals(data):
    s = st.signals(data)
    assert (s["flat"] == 0).all() and (s["long"] == 1).all()
    assert (s["carry"] == np.sign(data["rate_diff"])).all()
    assert (s["lightgbm"] == np.sign(data["p_lightgbm"] - 0.5)).all()
    confident = s["lightgbm_confident"]
    assert set(confident.unique()) <= {-1.0, 0.0, 1.0}
    later = confident.iloc[100:]
    assert 0.3 < (later != 0).mean() < 0.7  # roughly the more confident half of days


def test_confident_calls_use_only_earlier_forecasts(data):
    prob = data["p_lightgbm"].copy()
    before = st.confident(prob)
    prob.iloc[300:] = 0.5 + 10 * (prob.iloc[300:] - 0.5)
    after = st.confident(prob)
    pd.testing.assert_series_equal(before.iloc[:300], after.iloc[:300])


def test_volatility_targeting(data):
    size = st.sizes(data)
    expected = (st.TARGET_VOL / np.sqrt(data["var_baseline_har"] * 252)).clip(upper=3)
    pd.testing.assert_series_equal(size["vt_har"], expected, check_names=False)
    assert (size["unit"] == 1).all()
    assert size.max().max() <= st.MAX_LEVERAGE


def test_strategy_returns_charge_half_spread_per_unit_traded():
    dates = pd.bdate_range("2020-01-01", periods=4)
    data = pd.DataFrame({"ret": [0.01, -0.02, 0.03, 0.0], "carry": [0.001, 0.0, 0.0, 0.0],
                         "half_spread": 0.0001}, index=dates)
    signal = pd.Series([1.0, -1.0, -1.0, 0.0], index=dates)
    out = st.strategy_returns(data, signal, pd.Series(2.0, index=dates))
    assert out["turnover"].tolist() == [2.0, 4.0, 0.0, 2.0]
    assert out["gross"].tolist() == pytest.approx([0.022, 0.04, -0.06, 0.0])
    assert out["cost"].tolist() == pytest.approx([0.0002, 0.0004, 0.0, 0.0002])
    doubled = st.strategy_returns(data, signal, pd.Series(2.0, index=dates), 2.0)
    assert doubled["cost"].sum() == pytest.approx(2 * out["cost"].sum())


def test_run_pair_covers_every_strategy_once(data):
    results = st.run_pair(data, (1.0, 2.0))
    strategies = results[["signal", "sizing"]].drop_duplicates()
    n_signals = 4 + 2 * len(st.DIRECTION_MODELS)
    assert len(strategies) == 1 + (n_signals - 1) * (1 + len(st.SIZING_MODELS))
    assert "net_2x" in results
    flat = results[results["signal"] == "flat"]
    assert (flat["net"] == 0).all()


def test_performance_summary():
    dates = pd.bdate_range("2020-01-01", periods=5)
    daily = pd.DataFrame({"position": 1.0, "turnover": [1.0, 0, 0, 0, 0],
                          "cost": [0.001, 0, 0, 0, 0],
                          "net": [0.01, -0.03, 0.02, -0.01, 0.005]}, index=dates)
    out = perf.summary(daily)
    assert out["ann_return"] == pytest.approx(np.mean(daily["net"]) * 252)
    assert out["max_drawdown"] == pytest.approx(-0.03)  # from the 0.01 peak down to -0.02
    assert out["hit_rate"] == pytest.approx(0.6)
    assert out["turnover"] == pytest.approx(0.2 * 252)


def test_newey_west_matches_the_plain_t_test_without_autocorrelation():
    x = np.random.default_rng(1).normal(0.05, 1, 20_000)
    t, _ = perf.newey_west_t(x)
    plain = stats.ttest_1samp(x, 0).statistic
    assert t == pytest.approx(plain, rel=0.05)


def test_portfolio_averages_pairs():
    rows = []
    for pair, net in [("A", 0.02), ("B", -0.01)]:
        rows.append(pd.DataFrame({"signal": "long", "sizing": "unit", "pair": pair,
                                  "date": pd.bdate_range("2020-01-01", periods=3),
                                  "position": 1.0, "turnover": 0.0, "gross": net, "cost": 0.0,
                                  "net": net, "net_2x": net}))
    out = perf.portfolio(pd.concat(rows))
    assert out["net"].tolist() == pytest.approx([0.005] * 3)


def test_risk_targeting_rewards_accurate_volatility_forecasts():
    frame, close, predictions, variance = toy_inputs(n=1500, seed=4)
    exact = predictions.copy()
    volatility = exact["task"] == "volatility"
    exact.loc[volatility & (exact["model"] == "lightgbm"), "forecast"] = variance  # perfect
    exact.loc[volatility & (exact["model"] == "baseline_naive"), "forecast"] = variance.mean()
    data = st.pair_data(frame, close, exact, spread=0.0)
    results = st.run_pair(data).assign(pair="X")
    table = perf.risk_targeting(results, st.TARGET_VOL).set_index("sizing")
    assert table.loc["vt_lightgbm", "tracking"] < table.loc["vt_naive", "tracking"]
    assert table.loc["vt_lightgbm", "mean_realised_vol"] == pytest.approx(0.10, rel=0.1)


def test_value_at_risk_normal_and_historical():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2012-01-02", periods=3000)
    sigma = 0.006 * np.exp(rng.normal(0, 0.3, len(dates)))
    returns = pd.Series(rng.standard_t(4, len(dates)) / np.sqrt(2) * sigma, index=dates)
    variance = pd.Series(sigma ** 2, index=dates)
    var = risk.value_at_risk(variance, returns)
    assert var["normal_long"].to_numpy() == pytest.approx(stats.norm.ppf(0.99) * sigma)
    assert var["fhs_long"].iloc[: risk.FHS_MIN_DAYS].isna().all()
    hits = risk.exceedances(var, returns)
    later = hits.iloc[risk.FHS_WINDOW:]
    # fat tails: the normal Value at Risk is beaten too often, historical simulation is close
    assert later["normal_long"].mean() > 0.012
    assert abs(later["fhs_long"].mean() - 0.01) < 0.004
    out = risk.backtest_var(later["fhs_long"])
    assert out["days"] == len(later) and 0 <= out["kupiec_p"] <= 1


def test_historical_value_at_risk_uses_only_returns_already_known():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2012-01-02", periods=900)
    variance = pd.Series(1e-5, index=dates)
    returns = pd.Series(rng.normal(0, 0.003, len(dates)), index=dates)
    before = risk.value_at_risk(variance, returns)
    changed = returns.copy()
    changed.iloc[600:] *= 5
    after = risk.value_at_risk(variance, changed)
    pd.testing.assert_frame_equal(before.iloc[:601], after.iloc[:601])


def test_christoffersen_detects_clustered_exceedances():
    independent = np.zeros(2000, bool)
    independent[::100] = True
    clustered = np.zeros(2000, bool)
    clustered[1000:1020] = True
    assert christoffersen(independent)[1] > 0.5
    assert christoffersen(clustered)[1] < 0.001
    assert np.isnan(christoffersen(np.zeros(100, bool))[0])


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM cannot load here")
def test_shap_values_add_up_to_the_forecast(frame, folds):
    from fxml.interpret import contributions, yearly_shap
    from fxml.models import ml

    spec = ml.get_spec("volatility", "lightgbm")
    params = {"num_leaves": 7, "n_estimators": 50}
    model = ml.fit(spec, params, frame.iloc[folds[0].train])[None]
    x = frame.iloc[folds[0].test][ml.feature_columns(spec, frame)].to_numpy(dtype=float)
    full = model.predict(x, pred_contrib=True)
    np.testing.assert_allclose(full.sum(axis=1), model.predict(x), rtol=1e-6)
    assert contributions(model, x).shape == (len(x), x.shape[1])
    table = yearly_shap(spec, frame, folds[:2], {2012: '{"num_leaves": 7, "n_estimators": 50}'})
    assert set(table["year"]) == {2012}
    assert (table["mean_abs_shap"] >= 0).all()
    assert table.loc[table["feature"] == "garch_logvar", "group"].item() == "volatility"
