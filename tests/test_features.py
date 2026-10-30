import numpy as np
import pandas as pd
import pytest

from fxml.features import technical as ta
from fxml.features.build import FEATURES, build_table

from .conftest import PAIRS

CUTOFF = pd.Timestamp("2011-06-15")


def test_every_feature_and_target_is_present(synthetic_table):
    for column in [*FEATURES, "y_rv", "y_logrv", "y_ret", "y_dir", "target_date", "ret", "rv"]:
        assert column in synthetic_table.columns
    late = synthetic_table[synthetic_table.index.get_level_values("date") >= "2011-01-01"]
    assert late[FEATURES].notna().all().all()  # every feature filled once windows are warm


def test_features_never_use_the_future(synthetic_inputs, synthetic_table):
    """Scramble every input after CUTOFF; features up to CUTOFF must not change at all."""
    panel, macro, rates = synthetic_inputs
    rng = np.random.default_rng(1)
    after = panel.index.get_level_values("date") > CUTOFF
    scrambled = panel.copy()
    for col in ["open", "high", "low", "close", "noon", "rv", "rv_up", "rv_down",
                "parkinson", "garman_klass"]:
        scrambled.loc[after, col] *= rng.uniform(0.5, 1.5, after.sum())
    macro2 = macro.copy()
    macro2.loc[macro2.index > CUTOFF] *= 1.3
    rates2 = rates.copy()
    rates2.loc[rates2.index > CUTOFF] += 2.0

    rebuilt = build_table(scrambled, macro2, rates2, PAIRS)
    up_to = synthetic_table.index.get_level_values("date") <= CUTOFF
    pd.testing.assert_frame_equal(
        synthetic_table.loc[up_to, FEATURES], rebuilt.loc[up_to, FEATURES]
    )
    # ...while the target of the last row before the cut-off does see the change
    last = synthetic_table.loc[up_to].index[-1]
    assert synthetic_table.loc[last, "y_rv"] != rebuilt.loc[last, "y_rv"]


def test_targets_are_the_next_trading_day(synthetic_inputs, synthetic_table):
    panel, _, _ = synthetic_inputs
    daily = panel.xs("EURUSD", level="symbol")
    daily = daily[~daily["thin"]]
    table = synthetic_table.xs("EURUSD", level="pair")
    day, next_day = daily.index[100], daily.index[101]
    assert table.loc[day, "y_rv"] == daily.loc[next_day, "rv"]
    assert table.loc[day, "target_date"] == next_day
    assert table.loc[day, "y_ret"] == pytest.approx(
        np.log(daily.loc[next_day, "close"] / daily.loc[day, "close"]))


def test_holidays_are_dropped_and_returns_span_them(synthetic_inputs, synthetic_table):
    panel, _, _ = synthetic_inputs
    table = synthetic_table.xs("EURUSD", level="pair")
    christmas, before, after = (pd.Timestamp(d) for d in ("2009-12-25", "2009-12-24", "2009-12-28"))
    assert christmas not in table.index
    assert table.loc[before, "target_date"] == after
    close = panel.xs("EURUSD", level="symbol")["close"]
    assert table.loc[before, "y_ret"] == pytest.approx(np.log(close[after] / close[before]))


@pytest.mark.parametrize("indicator", [
    lambda o, hi, lo, c: ta.rsi(c),
    lambda o, hi, lo, c: ta.macd_histogram(c),
    lambda o, hi, lo, c: ta.bollinger_pct_b(c),
    lambda o, hi, lo, c: ta.atr_pct(hi, lo, c),
    lambda o, hi, lo, c: ta.stochastic(hi, lo, c)[1],
    lambda o, hi, lo, c: ta.adx(hi, lo, c),
])
def test_indicators_are_causal(synthetic_inputs, indicator):
    """An indicator's value on a day is the same whether or not later days exist."""
    daily = synthetic_inputs[0].xs("EURUSD", level="symbol")
    full = indicator(daily.open, daily.high, daily.low, daily.close)
    cut = daily.iloc[:400]
    prefix = indicator(cut.open, cut.high, cut.low, cut.close)
    pd.testing.assert_series_equal(full.iloc[:400], prefix, check_names=False)


def test_rsi_extremes():
    rising = pd.Series(np.linspace(1.0, 2.0, 50))
    assert ta.rsi(rising).iloc[-1] == pytest.approx(100)
    assert ta.rsi(rising[::-1].reset_index(drop=True)).iloc[-1] == pytest.approx(0)
