import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fxml.backtest import strategies as st
from fxml.config import _validate, load_config
from fxml.extra_pairs import (
    BAR_COLUMNS,
    INTRADAY_FEATURES,
    LAGGED_FEATURES,
    extra_features,
    extra_inputs,
    implied_h10,
    noon_bars,
)
from fxml.features.build import build_table
from fxml.models.inputs import pair_frame

from .conftest import FIRST_REFIT

# EURUSD stands in for an extra pair with intraday data, USDINR for a daily-only one.
CFG = SimpleNamespace(extra_pairs=("EURUSD", "USDINR"), daily_only={"USDINR": "DEXINUS"},
                      walk_forward=SimpleNamespace(test_start=FIRST_REFIT))
STATE_COLUMNS = ["pc1", "gmm_p1", "hmm_p1", "regime"]


@pytest.fixture(scope="module")
def extra(synthetic_inputs):
    panel, macro, rates = synthetic_inputs
    dates = panel.index.get_level_values("date").unique()
    rng = np.random.default_rng(11)
    noon = pd.Series(45 * np.exp(np.cumsum(rng.normal(0, 0.005, len(dates)))), index=dates)
    noon = noon.drop(dates[[50, 51, 400]])  # US holidays: no noon rate
    inr = noon_bars(noon).assign(symbol="USDINR").set_index("symbol", append=True)
    panel = pd.concat([panel, inr]).sort_index()
    rates = rates.assign(INR=6.5)
    return panel, macro, rates, extra_features(CFG, panel, macro, rates)


def test_noon_bars_use_the_noon_rate_as_the_close():
    noon = pd.Series([80.0, 81.0, np.nan, 80.5], index=pd.bdate_range("2020-01-06", periods=4))
    bars = noon_bars(noon)
    assert len(bars) == 3 and (bars["close"] == bars["noon"]).all()
    assert bars[BAR_COLUMNS].isna().all().all()
    assert bars["ret"].iloc[1] == pytest.approx(np.log(81 / 80))
    assert not bars["thin"].any() and not bars["partial"].any()


def test_daily_only_pairs_take_close_dated_inputs_from_the_day_before(extra):
    panel, macro, rates, table = extra
    plain = build_table(panel, macro, rates, ("USDINR",)).xs("USDINR", level="pair")
    inr = table.xs("USDINR", level="pair")
    pd.testing.assert_frame_equal(inr[LAGGED_FEATURES], plain[LAGGED_FEATURES].shift(1))
    assert (inr[INTRADAY_FEATURES] == 0).all().all()  # no intraday data: no information
    own = ["ret_lag1", "mom_21", "rsi_14", "ma_dist_50", "log_ret_std_21", "y_ret", "y_dir"]
    pd.testing.assert_frame_equal(inr[own], plain[own])  # its own noon prices are on time
    assert inr["y_rv"].isna().all()


def test_extra_pairs_with_intraday_data_match_the_core_pipeline(extra, synthetic_table):
    _, _, _, table = extra
    pd.testing.assert_frame_equal(table.xs("EURUSD", level="pair"),
                                  synthetic_table.xs("EURUSD", level="pair"))


def test_inputs_lag_the_market_state_for_daily_only_pairs(extra, state):
    _, _, _, table = extra
    inputs = extra_inputs(CFG, table, state)
    inr = inputs.xs("USDINR", level="pair")
    plain = pair_frame(table, "USDINR", state, FIRST_REFIT)
    pd.testing.assert_frame_equal(inr[STATE_COLUMNS], plain[STATE_COLUMNS].shift(1))
    pd.testing.assert_series_equal(inr["garch_logvar"], plain["garch_logvar"])  # own returns
    eur = inputs.xs("EURUSD", level="pair")
    pd.testing.assert_frame_equal(eur, pair_frame(table, "EURUSD", state, FIRST_REFIT),
                                  check_dtype=False)  # regime: float once a pair has gaps


def test_cross_rates_are_implied_from_the_dollar_legs():
    macro = pd.DataFrame({"h10_EURUSD": [1.2], "h10_GBPUSD": [1.5], "h10_USDJPY": [100.0]})
    assert implied_h10(macro, "EURGBP").iloc[0] == pytest.approx(0.8)
    assert implied_h10(macro, "EURJPY").iloc[0] == pytest.approx(120.0)


def test_sizing_skips_forecasts_a_pair_does_not_have():
    dates = pd.bdate_range("2020-01-01", periods=5)
    data = pd.DataFrame({"var_baseline_ewma": 1e-4, "var_baseline_garch": 2e-4}, index=dates)
    sizes = st.sizes(data)
    assert list(sizes.columns) == ["unit", "vt_ewma"]
    assert sizes["vt_ewma"].iloc[0] == pytest.approx(st.TARGET_VOL / np.sqrt(1e-4 * 252))


def test_config_lists_the_extra_pairs_with_their_costs():
    cfg = load_config()
    assert cfg.extra_pairs == ("EURGBP", "USDINR") and cfg.daily_only == {"USDINR": "DEXINUS"}
    assert cfg.costs.spread_in_price("USDINR") == pytest.approx(0.01)
    with pytest.raises(ValueError, match="repeat a core pair"):
        _validate(dataclasses.replace(cfg, extra_pairs=("EURUSD",), daily_only={}))
    with pytest.raises(ValueError, match="daily_only"):
        _validate(dataclasses.replace(cfg, daily_only={"USDTRY": "X"}))
