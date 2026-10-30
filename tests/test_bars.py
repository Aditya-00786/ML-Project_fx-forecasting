import numpy as np
import pandas as pd
import pytest

from fxml.data.bars import (
    add_daily_measures,
    clip_bad_wicks,
    daily_bars,
    resample_bars,
    trading_dates,
)

NY = "America/New_York"


def utc(stamp: str) -> pd.Timestamp:
    return pd.Timestamp(stamp, tz="UTC")


def test_resample_builds_ohlc_and_counts():
    index = pd.date_range("2026-01-05 15:00", periods=10, freq="1min", tz="UTC")
    m1 = pd.DataFrame(
        {"open": np.arange(10.0), "high": np.arange(10.0) + 0.5,
         "low": np.arange(10.0) - 0.5, "close": np.arange(10.0) + 0.25},
        index=index,
    )
    bars = resample_bars(m1.drop(index[7]), minutes=5)
    assert list(bars.index) == [utc("2026-01-05 15:00"), utc("2026-01-05 15:05")]
    first, second = bars.iloc[0], bars.iloc[1]
    assert (first.open, first.high, first.low, first.close, first.n) == (0.0, 4.5, -0.5, 4.25, 5)
    assert (second.open, second.close, second.n) == (5.0, 9.25, 4)


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-01-05 21:59", "2026-01-05"),  # 16:59 EST, winter
        ("2026-01-05 22:00", "2026-01-06"),  # 17:00 EST starts the next day
        ("2026-07-06 20:59", "2026-07-06"),  # 16:59 EDT, summer
        ("2026-07-06 21:00", "2026-07-07"),  # 17:00 EDT
        ("2026-01-04 22:30", "2026-01-05"),  # Sunday evening belongs to Monday
    ],
)
def test_trading_day_boundary_is_17h_new_york(stamp, expected):
    dates = trading_dates(pd.DatetimeIndex([utc(stamp)]), NY, close_hour=17)
    assert dates[0] == pd.Timestamp(expected)


def make_bars(stamps_and_closes):
    index = pd.DatetimeIndex([utc(s) for s, _ in stamps_and_closes])
    close = np.array([c for _, c in stamps_and_closes])
    return pd.DataFrame(
        {"open": close - 0.001, "high": close + 0.002, "low": close - 0.002,
         "close": close, "n": 5},
        index=index,
    )


def test_daily_bars_split_days_drop_weekends_and_sum_squared_returns():
    bars = make_bars([
        ("2026-01-04 22:00", 1.000),  # Sunday 17:00 EST -> Monday 5 Jan
        ("2026-01-05 16:55", 1.010),  # Monday 11:55 EST -> noon price for Monday
        ("2026-01-05 21:55", 1.020),  # Monday 16:55 EST -> Monday close
        ("2026-01-05 22:00", 1.030),  # Monday 17:00 EST -> Tuesday 6 Jan
        ("2026-01-09 22:30", 1.040),  # Friday 17:30 EST -> Saturday, dropped
    ])
    daily = daily_bars(bars, NY, close_hour=17)

    assert list(daily.index) == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    monday = daily.loc["2026-01-05"]
    assert monday.close == 1.020
    assert monday.noon == 1.010
    assert monday.n_bars == 3
    expected_rv = (
        np.log(1.000 / 0.999) ** 2 + np.log(1.010 / 1.000) ** 2 + np.log(1.020 / 1.010) ** 2
    )
    assert monday.rv == pytest.approx(expected_rv)
    assert np.isnan(daily.loc["2026-01-06"].noon)  # no bar before noon on Tuesday


def test_bad_tick_wick_is_removed_but_genuine_extreme_kept():
    index = pd.date_range("2008-12-23 10:00", periods=4, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {"open": [0.572, 0.572, 0.573, 0.900], "high": [0.573, 2.171, 0.574, 0.905],
         "low": [0.571, 0.572, 0.572, 0.870], "close": [0.572, 0.573, 0.573, 0.880]},
        index=index,
    )
    cleaned, changed = clip_bad_wicks(bars)
    assert changed == 1
    assert cleaned["high"].iloc[1] == 0.573  # cut back to the bar's body
    assert cleaned["low"].iloc[3] == 0.870   # a 1% wick in a fast market is left alone


def test_daily_measures_returns_and_thin_flag():
    daily = pd.DataFrame(
        {"open": [1.0, 1.1], "high": [1.2, 1.3], "low": [0.9, 1.0], "close": [1.1, 1.2],
         "n_minutes": [1400, 300]},
        index=pd.DatetimeIndex(["2026-01-05", "2026-01-06"], name="date"),
    )
    out = add_daily_measures(daily)
    assert out["ret"].iloc[1] == pytest.approx(np.log(1.2 / 1.1))
    assert out["parkinson"].iloc[0] == pytest.approx(np.log(1.2 / 0.9) ** 2 / (4 * np.log(2)))
    assert out["thin"].tolist() == [False, True]
