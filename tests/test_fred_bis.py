import zipfile
from datetime import date

import numpy as np
import pandas as pd

from fxml.data.bis import policy_rates, read_observations
from fxml.data.fred import parse_series

BIS_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,FREQ:Frequency,REF_AREA:Reference area,"
    "TIME_PERIOD:Time period or range,OBS_VALUE:Observation Value\n"
)


def bis_row(freq: str, area: str, period: str, value: str) -> str:
    return f"dataflow,BIS:WS_CBPOL(1.0),I,{freq},{area},{period},{value}\n"


def test_fred_missing_values_become_nan():
    series = parse_series("observation_date,VIXCLS\n2026-01-02,14.5\n2026-01-05,.\n2026-01-06,\n")
    assert series.loc["2026-01-02"] == 14.5
    assert series.isna().sum() == 2


def test_bis_daily_rates_with_monthly_fallback(tmp_path):
    rows = [
        bis_row("M: Monthly", "JP: Japan", "2005-12", "0.1"),
        bis_row("M: Monthly", "JP: Japan", "2006-01", "0.2"),
        bis_row("M: Monthly", "JP: Japan", "2006-03", "9.9"),  # after daily data starts: ignored
        bis_row("D: Daily", "JP: Japan", "2006-03-09", "0.5"),
        bis_row("D: Daily", "US: United States", "2005-12-30", "4.25"),
        bis_row("D: Daily", "US: United States", "2006-01-31", "NaN"),
        bis_row("D: Daily", "GB: United Kingdom", "2006-01-02", "4.5"),  # area not requested
    ]
    zip_path = tmp_path / "WS_CBPOL_csv_flat.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("WS_CBPOL_csv_flat.csv", BIS_HEADER + "".join(rows))

    obs = read_observations(zip_path, {"JP", "US"}, since=date(2005, 1, 1))
    rates = policy_rates(obs, {"JP": "JPY", "US": "USD"}, date(2006, 1, 1), date(2006, 3, 10))

    assert list(rates.columns) == ["JPY", "USD"]
    assert rates.loc["2006-01-15", "JPY"] == 0.1  # December value carried forward
    assert rates.loc["2006-02-15", "JPY"] == 0.2  # January month-end value
    assert rates.loc["2006-03-08", "JPY"] == 0.2
    assert rates.loc["2006-03-10", "JPY"] == 0.5  # daily data takes over
    assert rates.loc["2006-02-01", "USD"] == 4.25  # missing daily value does not erase the rate
    assert not np.isnan(rates.to_numpy()).any()
    assert rates.index[0] == pd.Timestamp("2006-01-01")


def test_bis_series_starting_late_is_carried_back(tmp_path):
    zip_path = tmp_path / "WS_CBPOL_csv_flat.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("WS_CBPOL_csv_flat.csv",
                         BIS_HEADER + bis_row("D: Daily", "JP: Japan", "2006-03-09", "0"))
    obs = read_observations(zip_path, {"JP"}, since=date(2005, 1, 1))
    rates = policy_rates(obs, {"JP": "JPY"}, date(2006, 1, 1), date(2006, 3, 10))
    assert rates["JPY"].notna().all()
    assert rates.loc["2006-01-01", "JPY"] == 0.0
