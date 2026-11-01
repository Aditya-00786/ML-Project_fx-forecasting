import numpy as np
import pytest

from fxml.validation.metrics import (
    direction_metrics,
    qlike_losses,
    return_metrics,
    volatility_metrics,
)
from fxml.validation.stats_tests import diebold_mariano, kupiec, pesaran_timmermann

rng = np.random.default_rng(0)


def test_qlike_is_zero_only_for_a_perfect_forecast_and_punishes_underprediction():
    realized = np.array([1.0, 2.0])
    assert qlike_losses(realized, realized) == pytest.approx([0, 0])
    assert qlike_losses(np.array([2.0]), np.array([1.0]))[0] > qlike_losses(
        np.array([1.0]), np.array([2.0]))[0]


def test_volatility_metrics_on_an_unbiased_forecast():
    forecast = np.exp(rng.normal(-9, 1, 5000))
    realized = forecast * np.exp(rng.normal(0, 0.3, 5000))
    m = volatility_metrics(realized, forecast)
    assert m["mz_slope"] == pytest.approx(1, abs=0.03)
    assert m["mz_r2"] > 0.85


def test_return_metrics_against_zero():
    actual = rng.normal(0, 1, 1000)
    assert return_metrics(actual, np.zeros(1000))["r2_oos"] == pytest.approx(0)
    assert return_metrics(actual, actual)["r2_oos"] == pytest.approx(1)


def test_direction_metrics():
    up = rng.integers(0, 2, 1000)
    m = direction_metrics(up, up.astype(float))
    assert m["accuracy"] == 1 and m["roc_auc"] == 1 and m["brier"] == 0


def test_diebold_mariano_detects_the_better_forecast_and_is_antisymmetric():
    better, worse = rng.normal(1.0, 0.2, 500), rng.normal(1.2, 0.2, 500)
    stat, p = diebold_mariano(better, worse)
    assert stat < 0 and p < 0.001
    assert diebold_mariano(worse, better)[0] == pytest.approx(-stat)
    _, p_same = diebold_mariano(rng.normal(1, 0.2, 500), rng.normal(1, 0.2, 500))
    assert p_same > 0.001


def test_pesaran_timmermann():
    actual = rng.integers(0, 2, 1000).astype(bool)
    stat, p = pesaran_timmermann(actual, actual)
    assert p < 1e-6
    _, p_random = pesaran_timmermann(actual, rng.integers(0, 2, 1000).astype(bool))
    assert p_random > 0.001
    assert np.isnan(pesaran_timmermann(actual, np.ones(1000, bool))[0])  # calls never change


def test_kupiec():
    right = np.zeros(1000, bool)
    right[::100] = True  # exactly 1%
    assert kupiec(right, 0.01)[1] > 0.99
    too_many = np.zeros(1000, bool)
    too_many[::25] = True  # 4%
    assert kupiec(too_many, 0.01)[1] < 0.001


def test_score_skips_days_a_model_did_not_forecast():
    import pandas as pd

    from fxml.validation.evaluate import score

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    realized = rng.lognormal(-9, 0.5, 300)
    frames = []
    noisy = {m: realized * rng.lognormal(0, 0.3, 300) for m in ("naive", "gappy")}
    for model, forecast in noisy.items():
        forecast = forecast.copy()
        if model == "gappy":
            forecast[10] = np.nan
        frames.append(pd.DataFrame({"task": "volatility", "pair": "EURUSD", "model": model,
                                    "date": dates, "actual": realized, "forecast": forecast}))
    scores = score(pd.concat(frames), {"volatility": "naive"}).set_index("model")
    assert scores.loc["gappy", "days"] == 299
    assert scores.loc["naive", "days"] == 300
    assert np.isfinite(scores.loc["gappy", ["qlike", "mz_r2", "dm_vs_naive"]].astype(float)).all()


def test_calibration_measures():
    from fxml.validation.metrics import (
        calibration_bins,
        calibration_slope,
        expected_calibration_error,
    )

    rng = np.random.default_rng(4)
    prob = rng.uniform(0.2, 0.8, 50_000)
    outcome = (rng.random(50_000) < prob).astype(float)  # calibrated by construction
    assert expected_calibration_error(outcome, prob) < 0.01
    slope, se = calibration_slope(outcome, prob)
    assert slope == pytest.approx(1.0, abs=0.05) and 0 < se < 0.05
    stretched = 1 / (1 + np.exp(-3 * np.log(prob / (1 - prob))))  # three times too confident
    assert expected_calibration_error(outcome, stretched) > 0.05
    assert calibration_slope(outcome, stretched)[0] == pytest.approx(1 / 3, abs=0.03)
    # forecasts that barely move: the slope is poorly identified, and its error says so
    flat = 0.5 + 0.005 * (prob - 0.5)
    assert calibration_slope(outcome, flat)[1] > 10 * se
    table = calibration_bins(outcome, prob, bins=5)
    assert len(table) == 5 and table["days"].sum() == 50_000
    assert np.isnan(calibration_slope(outcome, np.full(50_000, 0.5))).all()  # constant
    assert len(calibration_bins(outcome, np.full(50_000, 0.5))) == 1
    # a forecast that is always 0 or 1 on coin flips is badly calibrated, whatever the binning
    coin = (rng.random(50_000) < 0.5).astype(float)
    certain = (rng.random(50_000) < 0.5).astype(float)
    assert len(calibration_bins(coin, certain)) == 2
    assert expected_calibration_error(coin, certain) == pytest.approx(0.5, abs=0.02)
