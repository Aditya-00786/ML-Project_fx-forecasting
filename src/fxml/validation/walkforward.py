"""Expanding-window walk-forward folds with purging and an embargo.

Each fold tests one block (a month by default). Its training rows are every earlier row whose
*target* is already known before the block starts. A row's target is the next trading day, so
the last row before the block is purged, because its label falls inside the block. Tuning uses
the last part of the training rows as a validation slice. Rows just before that slice are
embargoed out of the fitting rows, so they can't share information with it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    test_start: pd.Timestamp
    train: np.ndarray  # positions usable for training once tuning is done (fit + embargo + valid)
    fit: np.ndarray  # positions for fitting during tuning
    valid: np.ndarray  # positions for scoring during tuning
    test: np.ndarray  # positions to forecast


def walk_forward(
    dates: pd.DatetimeIndex,
    target_dates: pd.Series | pd.DatetimeIndex,
    test_start,
    freq: str = "MS",
    validation_fraction: float = 0.2,
    embargo: int = 5,
) -> Iterator[Fold]:
    """Yield folds over ``dates`` (sorted, one row per date) from ``test_start`` onwards.

    ``target_dates`` gives, for each row, the date its target is observed. Rows whose target is
    missing (the last day of the data) are never used for training or testing.
    """
    dates = pd.DatetimeIndex(dates)
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dates must be sorted and unique")
    target = pd.DatetimeIndex(target_dates)
    has_target = ~target.isna()
    positions = np.arange(len(dates))

    starts = pd.date_range(pd.Timestamp(test_start), dates.max(), freq=freq)
    for start, stop in zip(starts, [*starts[1:], pd.Timestamp.max], strict=True):
        test = positions[(dates >= start) & (dates < stop) & has_target]
        if len(test) == 0:
            continue
        train = positions[(dates < start) & has_target & (target < start)]
        n_valid = int(round(len(train) * validation_fraction))
        valid = train[len(train) - n_valid :]
        fit = train[: max(len(train) - n_valid - embargo, 0)]
        yield Fold(test_start=start, train=train, fit=fit, valid=valid, test=test)
