from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fxml.models import neural
from fxml.models.comparison import seed_average
from fxml.models.inputs import BASE_FEATURES

from .conftest import CUT


def quick(spec: neural.NeuralSpec) -> neural.NeuralSpec:
    """The same network, trained briefly so tests stay fast."""
    return replace(spec, max_epochs=4, patience=2)


def test_windows_only_look_back():
    z = np.arange(12, dtype=float).reshape(6, 2)
    w = neural.windows(z, 3)
    assert w.shape == (6, 3, 2)
    np.testing.assert_array_equal(w[4], z[2:5])  # the window for row 4 ends at row 4
    np.testing.assert_array_equal(w[0], np.repeat(z[:1], 3, axis=0))  # history padded with day 0
    np.testing.assert_array_equal(w[1], z[[0, 0, 1]])


def test_scaling_uses_training_rows_only(frame, folds):
    train = folds[0].fit
    first = neural.prepare(frame, "volatility", train, 1)
    changed = frame.copy()
    later = np.setdiff1d(np.arange(len(frame)), train)
    changed.iloc[later, changed.columns.get_indexer(BASE_FEATURES)] *= 3.0
    changed.iloc[later, changed.columns.get_loc("y_logrv")] += 5.0
    second = neural.prepare(changed, "volatility", train, 1)
    np.testing.assert_array_equal(first.x[train], second.x[train])
    assert (first.centre, first.spread) == (second.centre, second.spread)


@pytest.mark.parametrize("spec", neural.NEURAL_SPECS, ids=lambda s: f"{s.task}/{s.name}")
def test_every_network_forecasts_every_test_day(spec, frame, folds):
    few = folds[:2]  # January and February: one quarterly refit
    forecast, log = neural.run_neural_spec(quick(spec), frame, few, seed=1)
    expected = np.concatenate([fold.test for fold in few])
    assert forecast.index.equals(frame.index[expected])
    assert np.isfinite(forecast).all()
    if spec.task == "volatility":
        assert (forecast > 0).all()
    if spec.task == "direction":
        assert ((forecast > 0) & (forecast < 1)).all()
    assert len(log) == 1
    assert 1 <= log["epochs"].iloc[0] <= 4 and log["scale"].iloc[0] > 0


def test_refits_once_a_quarter(frame, folds):
    spec = quick(neural.get_neural_spec("returns", "mlp"))
    _, log = neural.run_neural_spec(spec, frame, folds[:6], seed=0)
    assert log["period"].tolist() == ["2012Q1", "2012Q2"]
    _, monthly = neural.run_neural_spec(spec, frame, folds[:3], seed=0, refit="MS")
    assert len(monthly) == 3


@pytest.mark.parametrize("task, name", [("volatility", "lstm"), ("direction", "mlp")])
def test_networks_never_use_the_future(task, name, frame, folds):
    """Scrambling every day after CUT leaves forecasts up to CUT unchanged."""
    spec = quick(neural.get_neural_spec(task, name))
    some = [f for f in folds if f.test_start <= CUT]
    before, _ = neural.run_neural_spec(spec, frame, some, seed=3)
    rng = np.random.default_rng(0)
    scrambled = frame.copy()
    later = scrambled.index > CUT
    columns = [*BASE_FEATURES, "y_rv", "y_logrv", "y_ret"]
    scrambled.loc[later, columns] *= rng.uniform(0.5, 1.5, (later.sum(), len(columns)))
    scrambled.loc[later, "y_dir"] = 1 - scrambled.loc[later, "y_dir"]
    after, _ = neural.run_neural_spec(spec, scrambled, some, seed=3)
    keep = before.index <= CUT
    pd.testing.assert_series_equal(before[keep], after[keep])


def test_same_seed_same_forecasts_and_seeds_average(frame, folds):
    spec = quick(neural.get_neural_spec("volatility", "mlp"))
    one, _ = neural.run_neural_spec(spec, frame, folds[:1], seed=5)
    again, _ = neural.run_neural_spec(spec, frame, folds[:1], seed=5)
    other, _ = neural.run_neural_spec(spec, frame, folds[:1], seed=6)
    pd.testing.assert_series_equal(one, again)
    assert not np.allclose(one, other)
    rows = pd.concat([
        pd.DataFrame({"task": "volatility", "pair": "EURUSD", "model": "mlp", "seed": s,
                      "date": f.index, "actual": 1.0, "forecast": f.to_numpy()})
        for s, f in [(0, one), (1, other)]
    ])
    averaged = seed_average(rows).set_index("date")["forecast"]
    np.testing.assert_allclose(averaged.to_numpy(), ((one + other) / 2).to_numpy())
