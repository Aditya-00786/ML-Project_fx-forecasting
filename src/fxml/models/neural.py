"""Neural networks for the three tasks: a multilayer perceptron (MLP) on each day's features and
an LSTM over a window of recent days.

At each refit (every quarter by default):

1. Inputs are imputed with the median and standardised using the training rows only. Volatility
   and return targets are standardised the same way.
2. The network is trained on the fold's fitting rows with Adam, and early stopping on the
   validation slice picks the epoch with the lowest validation loss.
3. A fresh network is trained on all training rows for that many epochs, so the most recent
   days are used too.

Mini-batches are drawn in random order within the training rows. Splits themselves always
respect time order. Volatility forecasts get the same scale factor as the classical models,
fitted on the validation forecasts of step 2. Each model is trained with several seeds;
``scripts/run_neural.py`` averages their forecasts.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn

from fxml.models.inputs import BASE_FEATURES
from fxml.models.ml import TARGETS, to_forecast, validation_loss
from fxml.validation.walkforward import Fold, refit_periods

CLIP = 10.0  # standardised inputs are clipped to +-10 to tame rare extreme days


@dataclass(frozen=True)
class NeuralSpec:
    name: str
    task: str
    window: int = 1  # trading days of history the network sees (1 = today only)
    hidden: int = 64
    dropout: float = 0.1
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


TASKS = ("volatility", "direction", "returns")
NEURAL_SPECS: list[NeuralSpec] = [
    *(NeuralSpec("mlp", task) for task in TASKS),
    *(NeuralSpec("lstm", task, window=20, hidden=32) for task in TASKS),
]


def get_neural_spec(task: str, name: str) -> NeuralSpec:
    return next(s for s in NEURAL_SPECS if s.task == task and s.name == name)


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, -1, :]).squeeze(-1)  # only the latest day


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build(spec: NeuralSpec, n_features: int) -> nn.Module:
    network = MLP if spec.name == "mlp" else LSTMNet
    return network(n_features, spec.hidden, spec.dropout)


# ---------------------------------------------------------------- data


@dataclass
class Prepared:
    x: np.ndarray  # (rows, window, features), standardised
    y: np.ndarray  # standardised target (raw 0/1 for direction)
    centre: float
    spread: float


def windows(z: np.ndarray, window: int) -> np.ndarray:
    """For every row, the ``window`` rows ending at it. The first rows repeat the first day in
    place of the history they lack, so each window only ever looks back."""
    padded = np.concatenate([np.repeat(z[:1], window - 1, axis=0), z])
    view = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)  # (rows, F, window)
    return np.ascontiguousarray(view.transpose(0, 2, 1), dtype=np.float32)


def prepare(frame: pd.DataFrame, task: str, train: np.ndarray, window: int) -> Prepared:
    """Inputs and target for every row, scaled with statistics of the ``train`` rows only."""
    raw = frame[BASE_FEATURES].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(raw[train])
    scaler = StandardScaler().fit(imputer.transform(raw[train]))
    z = np.clip(scaler.transform(imputer.transform(raw)), -CLIP, CLIP)
    y = frame[TARGETS[task]].to_numpy(dtype=float)
    centre, spread = 0.0, 1.0
    if task != "direction":
        centre, spread = float(np.nanmean(y[train])), float(np.nanstd(y[train]))
    return Prepared(windows(z, window), ((y - centre) / spread).astype(np.float32), centre, spread)


# ---------------------------------------------------------------- training


def _loss(task: str):
    return nn.BCEWithLogitsLoss() if task == "direction" else nn.MSELoss()


def train(
    spec: NeuralSpec, data: Prepared, rows: np.ndarray, seed: int,
    valid: np.ndarray | None = None, epochs: int | None = None,
) -> tuple[nn.Module, int]:
    """Train on ``rows``. With ``valid``, stop early and restore the best epoch's weights;
    otherwise train for ``epochs``. Returns the network and the number of epochs used.

    If no epoch gives a finite validation loss (or the slice has no labelled rows), the weights
    after the last epoch are kept and that epoch count is returned."""
    torch.manual_seed(seed)
    order = np.random.default_rng(seed)
    rows = rows[~np.isnan(data.y[rows])]
    x, y = torch.from_numpy(data.x[rows]), torch.from_numpy(data.y[rows])
    model = build(spec, data.x.shape[2])
    optimiser = torch.optim.Adam(model.parameters(), lr=spec.learning_rate,
                                 weight_decay=spec.weight_decay)
    loss_fn = _loss(spec.task)
    if valid is not None:
        valid = valid[~np.isnan(data.y[valid])]
        if len(valid) == 0:
            valid = None
        else:
            x_valid, y_valid = torch.from_numpy(data.x[valid]), torch.from_numpy(data.y[valid])
    best, best_state, best_epoch, waited, epoch = np.inf, None, 1, 0, 0
    for epoch in range(1, (epochs or spec.max_epochs) + 1):
        model.train()
        shuffled = order.permutation(len(rows))
        for start in range(0, len(rows), spec.batch_size):
            batch = shuffled[start : start + spec.batch_size]
            optimiser.zero_grad()
            loss_fn(model(x[batch]), y[batch]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        if valid is None:
            continue
        model.eval()
        with torch.no_grad():
            current = loss_fn(model(x_valid), y_valid).item()
        if current < best - 1e-6:
            best, best_epoch, waited = current, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= spec.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        return model, best_epoch
    return model, epoch


def predict(spec: NeuralSpec, model: nn.Module, data: Prepared, rows: np.ndarray) -> np.ndarray:
    """Raw predictions: log variance, probability of an up day, or return."""
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(data.x[rows]))
        if spec.task == "direction":
            return torch.sigmoid(out).numpy().astype(float)
    return out.numpy().astype(float) * data.spread + data.centre


def run_neural_spec(
    spec: NeuralSpec, frame: pd.DataFrame, folds: Sequence[Fold], seed: int = 42,
    refit: str = "QS",
) -> tuple[pd.Series, pd.DataFrame]:
    """Forecasts for every test row of ``folds`` with one seed, and a log of each refit.

    The network is refitted at the first fold and then at each ``refit`` boundary (any pandas
    frequency, quarterly by default)."""
    torch.set_num_threads(1)  # parallelism comes from running jobs side by side
    starts = [s for s, _ in refit_periods(folds[0].test_start, folds[-1].test_start, refit)]
    forecasts, log, period, model, data, scale = [], [], None, None, None, 1.0
    for fold in folds:
        key = max(s for s in starts if s <= fold.test_start)
        if key != period:
            period = key
            first = prepare(frame, spec.task, fold.fit, spec.window)
            searched, epochs = train(spec, first, fold.fit, seed, valid=fold.valid)
            valid_rows = frame.iloc[fold.valid]
            keep = valid_rows[TARGETS[spec.task]].notna().to_numpy()
            raw = predict(spec, searched, first, fold.valid[keep])
            loss, scale = validation_loss(spec.task, raw, valid_rows[keep])
            data = prepare(frame, spec.task, fold.train, spec.window)
            model, _ = train(spec, data, fold.train, seed, epochs=epochs)
            log.append({"period": f"{key:%Y-%m-%d}", "epochs": epochs, "validation_loss": loss,
                        "scale": scale})
        raw = predict(spec, model, data, fold.test)
        forecasts.append(pd.Series(to_forecast(spec.task, raw, scale),
                                   index=frame.index[fold.test]))
    return pd.concat(forecasts), pd.DataFrame(log)
