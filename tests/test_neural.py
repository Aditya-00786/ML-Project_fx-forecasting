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
    assert log["period"].tolist() == ["2012-01-01", "2012-04-01"]
    _, monthly = neural.run_neural_spec(spec, frame, folds[:3], seed=0, refit="MS")
    assert len(monthly) == 3
    _, yearly = neural.run_neural_spec(spec, frame, folds[:6], seed=0, refit="YS")
    assert yearly["period"].tolist() == ["2012-01-01"]
    _, fortnightly = neural.run_neural_spec(spec, frame, folds[:2], seed=0, refit="2W")
    assert len(fortnightly) == 2  # any pandas frequency works; each monthly fold refits once


def test_training_survives_a_validation_loss_that_is_never_finite(frame, folds):
    spec = quick(neural.get_neural_spec("volatility", "mlp"))
    data = neural.prepare(frame, "volatility", folds[0].fit, 1)
    broken = neural.Prepared(data.x, data.y.copy(), data.centre, data.spread)
    broken.y[folds[0].valid] = np.inf  # every validation loss is infinite
    model, epochs = neural.train(spec, broken, folds[0].fit, seed=0, valid=folds[0].valid)
    assert epochs == 2  # patience ran out after two epochs without a finite improvement
    unlabelled = neural.Prepared(data.x, data.y.copy(), data.centre, data.spread)
    unlabelled.y[folds[0].valid] = np.nan
    _, epochs = neural.train(spec, unlabelled, folds[0].fit, seed=0, valid=folds[0].valid)
    assert epochs == spec.max_epochs  # no labelled validation rows: train the full budget


@pytest.mark.parametrize("task, name", [
    ("volatility", "lstm"), ("direction", "mlp"), ("returns", "transformer"),
])
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


def test_cache_drops_jobs_whose_signature_changed(tmp_path, frame):
    from types import SimpleNamespace

    from fxml.models.comparison import read_cache, signatures, sync_cache, write_job

    cfg = SimpleNamespace(walk_forward="wf", seed=1)
    mlp, lstm = (neural.get_neural_spec("volatility", n) for n in ("mlp", "lstm"))
    first = signatures([mlp, lstm], frame, cfg, neural)
    jobs, log = pd.DataFrame({"forecast": [1.0]}), pd.DataFrame({"epochs": [3]})
    for prefix in first:
        write_job(tmp_path / f"{prefix}__EURUSD__seed0.parquet", jobs, log)
    assert sync_cache(tmp_path, first) == []
    changed = signatures([replace(mlp, hidden=128), lstm], frame, cfg, neural)
    assert sync_cache(tmp_path, changed) == ["volatility__mlp"]
    remaining = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert remaining == ["volatility__lstm__EURUSD__seed0.log.parquet",
                         "volatility__lstm__EURUSD__seed0.parquet"]
    other_inputs = signatures([lstm], frame.assign(ret=frame["ret"] * 2), cfg, neural)
    assert other_inputs != {k: v for k, v in changed.items() if "lstm" in k}
    # a log whose forecasts never appeared (an interrupted job) is ignored
    (tmp_path / "volatility__lstm__GBPUSD__seed0.log.parquet").write_bytes(
        (tmp_path / "volatility__lstm__EURUSD__seed0.log.parquet").read_bytes())
    forecasts, logs = read_cache(tmp_path)
    assert len(forecasts) == 1 and len(logs) == 1
