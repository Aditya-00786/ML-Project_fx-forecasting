import numpy as np
import pandas as pd
import pytest

from fxml.data.quality import coverage_report, expected_trading_days, h10_check
from fxml.events import stress_label


def test_expected_days_skip_weekends_christmas_and_new_year():
    days = expected_trading_days(pd.Timestamp("2025-12-22"), pd.Timestamp("2026-01-02"))
    assert pd.Timestamp("2025-12-25") not in days
    assert pd.Timestamp("2026-01-01") not in days
    assert pd.Timestamp("2025-12-27") not in days  # Saturday
    assert len(days) == 8


def panel_for(dates, noon):
    index = pd.MultiIndex.from_product([pd.DatetimeIndex(dates, name="date"), ["EURUSD"]],
                                       names=["date", "symbol"])
    return pd.DataFrame({"noon": noon, "thin": False, "n_minutes": 1440}, index=index)


def test_coverage_counts_missing_days():
    dates = pd.bdate_range("2026-01-05", "2026-01-16").delete(3)  # drop Thursday 8 Jan
    report = coverage_report(panel_for(dates, 1.0))
    assert report.loc["EURUSD", "missing"] == 1
    assert report.loc["EURUSD", "coverage_pct"] == pytest.approx(90.0)


def test_h10_check_detects_identical_series():
    dates = pd.bdate_range("2026-01-05", periods=30)
    prices = 1.1 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.005, 30)))
    macro = pd.DataFrame({"h10_EURUSD": prices}, index=dates)
    result = h10_check(panel_for(dates, prices), macro, ("EURUSD",))
    assert result.loc["EURUSD", "return_corr"] == pytest.approx(1.0)
    assert result.loc["EURUSD", "mean_abs_level_gap_pct"] == pytest.approx(0.0)


def test_stress_labels():
    labels = stress_label(pd.DatetimeIndex(["2015-01-15", "2019-06-03"]))
    assert labels.tolist() == ["SNB removes franc cap", "Normal"]
