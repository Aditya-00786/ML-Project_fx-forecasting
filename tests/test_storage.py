from dataclasses import replace

import pandas as pd

from fxml.config import load_config
from fxml.data.storage import bars_dir, load_bars


def test_full_year_file_wins_over_leftover_month_files(tmp_path):
    cfg = load_config()
    cfg = replace(cfg, paths=replace(cfg.paths, interim=tmp_path))
    folder = bars_dir(cfg, "EURUSD")
    folder.mkdir(parents=True)
    times = pd.date_range("2026-10-01", periods=3, freq="5min", tz="UTC")
    year = pd.DataFrame({"close": [1.0, 1.1, 1.2]}, index=times)
    month = pd.DataFrame({"close": [9.0, 9.0]}, index=times[:2])  # stale partial month
    year.to_parquet(folder / "2026.parquet")
    month.to_parquet(folder / "2026-10.parquet")
    other = pd.DataFrame({"close": [2.0]}, index=pd.DatetimeIndex(["2027-01-04"], tz="UTC"))
    other.to_parquet(folder / "2027-01.parquet")  # month of a year without a full-year file

    bars = load_bars(cfg, "EURUSD")
    assert bars["close"].tolist() == [1.0, 1.1, 1.2, 2.0]
