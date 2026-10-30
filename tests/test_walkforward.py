import numpy as np
import pandas as pd
import pytest

from fxml.validation.walkforward import walk_forward


@pytest.fixture
def calendar():
    dates = pd.bdate_range("2010-01-01", "2011-06-30")
    target = pd.Series(dates, index=dates).shift(-1)  # next trading day; the last row has none
    return dates, target


def test_folds_never_train_on_labels_from_the_test_period(calendar):
    dates, target = calendar
    folds = list(walk_forward(dates, target, "2011-01-01", embargo=5))
    assert [f.test_start.month for f in folds] == [1, 2, 3, 4, 5, 6]
    for fold in folds:
        assert dates[fold.train].max() < fold.test_start
        assert pd.DatetimeIndex(target.iloc[fold.train]).max() < fold.test_start  # purged
        assert dates[fold.test].min() >= fold.test_start
        assert set(fold.train).isdisjoint(fold.test)


def test_purging_drops_the_row_whose_label_is_the_first_test_day(calendar):
    dates, target = calendar
    fold = next(walk_forward(dates, target, "2011-02-01"))
    last_january = dates[(dates < "2011-02-01")][-1]
    assert dates.get_loc(last_january) not in fold.train
    assert dates.get_loc(last_january) - 1 in fold.train


def test_validation_is_the_latest_slice_after_an_embargo(calendar):
    dates, target = calendar
    fold = next(walk_forward(dates, target, "2011-01-01", validation_fraction=0.2, embargo=5))
    assert len(fold.valid) == round(0.2 * len(fold.train))
    assert fold.valid[-1] == fold.train[-1]
    assert fold.fit.max() + 5 + 1 == fold.valid.min()  # five rows embargoed in between


def test_window_expands_and_rows_without_targets_are_skipped(calendar):
    dates, target = calendar
    folds = list(walk_forward(dates, target, "2011-01-01"))
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert len(dates) - 1 not in np.concatenate([f.test for f in folds])  # last row: no target


def test_unsorted_dates_are_rejected():
    dates = pd.DatetimeIndex(["2011-01-04", "2011-01-03"])
    with pytest.raises(ValueError):
        list(walk_forward(dates, dates, "2011-01-01"))


def test_refit_periods_start_at_the_first_date():
    from fxml.validation.walkforward import refit_periods

    periods = refit_periods("2012-03-15", "2014-06-30", "YS")
    assert [str(s.date()) for s, _ in periods] == ["2012-03-15", "2013-01-01", "2014-01-01"]
    assert periods[0][1] == pd.Timestamp("2013-01-01")
    assert periods[-1][1] == pd.Timestamp.max
    assert [s for s, _ in refit_periods("2012-01-01", "2012-03-31", "MS")] == list(
        pd.date_range("2012-01-01", "2012-03-01", freq="MS"))
