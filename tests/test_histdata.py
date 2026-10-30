import io
import zipfile
from datetime import date

import pandas as pd

from fxml.data.histdata import (
    Period,
    clean_m1,
    parse_form_fields,
    parse_m1_csv,
    parse_period_links,
    period_complete,
    read_zip,
    select_periods,
)

SAMPLE_CSV = (
    "20260101 170400;1.173870;1.174140;1.173870;1.174140;0\n"
    "20260101 170500;1.174140;1.174490;1.174140;1.174480;0\n"
)


def test_period_links_are_parsed_and_sorted():
    base = "/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd"
    html = f'<a href="{base}/2026/10">x</a><a href="{base}/2024">y</a><a href="{base}/2026/1">z</a>'
    assert parse_period_links(html, "EURUSD") == [Period(2024), Period(2026, 1), Period(2026, 10)]


def test_select_periods_keeps_window_and_drops_redundant_months():
    offered = [Period(2005), Period(2006), Period(2025), Period(2025, 3), Period(2026, 9),
               Period(2026, 10), Period(2026, 11)]
    picked = select_periods(offered, date(2006, 1, 1), date(2026, 10, 30))
    assert picked == [Period(2006), Period(2025), Period(2026, 9), Period(2026, 10)]


def test_form_fields_are_extracted():
    html = "".join(
        f'<input type="hidden" name="{n}" id="{n}" value="{v}" />'
        for n, v in [("tk", "abc"), ("date", "2026"), ("datemonth", "202601"),
                     ("platform", "ASCII"), ("timeframe", "M1"), ("fxpair", "EURUSD")]
    )
    assert parse_form_fields(html)["datemonth"] == "202601"


def utc_of(stamp: str) -> pd.Timestamp:
    return parse_m1_csv(f"{stamp};1.1;1.2;1.0;1.15;0\n").index[0]


def test_clock_is_new_york_time_until_2018():
    assert utc_of("20150105 120000") == pd.Timestamp("2015-01-05 17:00", tz="UTC")  # EST
    assert utc_of("20150706 120000") == pd.Timestamp("2015-07-06 16:00", tz="UTC")  # EDT
    # 9 March 2015: US on daylight saving, EU not yet; the clock still follows New York
    assert utc_of("20150309 120000") == pd.Timestamp("2015-03-09 16:00", tz="UTC")


def test_clock_is_eet_minus_seven_hours_from_2019():
    assert utc_of("20260105 120000") == pd.Timestamp("2026-01-05 17:00", tz="UTC")
    assert utc_of("20260706 120000") == pd.Timestamp("2026-07-06 16:00", tz="UTC")
    # 9 March 2026: US on daylight saving, EU not yet; the clock is an hour behind New York
    assert utc_of("20260309 120000") == pd.Timestamp("2026-03-09 17:00", tz="UTC")
    assert list(parse_m1_csv(SAMPLE_CSV).columns) == ["open", "high", "low", "close"]


def test_clock_times_skipped_by_daylight_saving_are_dropped():
    text = (
        "20150308 023000;1.1;1.2;1.0;1.15;0\n"  # does not exist in New York
        "20260328 203000;1.1;1.2;1.0;1.15;0\n"  # 03:30 in Athens on 29 March 2026: does not exist
        "20260329 170000;1.1;1.2;1.0;1.15;0\n"
    )
    assert len(parse_m1_csv(text)) == 1


def test_partial_month_is_not_complete():
    october = Period(2026, 10)
    end = date(2026, 10, 30)
    assert not period_complete(october, pd.Timestamp("2026-10-23 20:58", tz="UTC"), end)
    assert period_complete(october, pd.Timestamp("2026-10-30 20:58", tz="UTC"), end)
    assert period_complete(Period(2015), pd.Timestamp("2015-12-31 21:59", tz="UTC"), end)
    assert not period_complete(Period(2015), pd.Timestamp("2015-06-30 21:59", tz="UTC"), end)


def test_zip_is_read_in_memory():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("DAT_ASCII_EURUSD_M1_202601.csv", SAMPLE_CSV)
        archive.writestr("DAT_ASCII_EURUSD_M1_202601.txt", "status report")
    assert len(read_zip(buffer.getvalue())) == 2


def test_duplicates_and_impossible_bars_are_dropped():
    text = SAMPLE_CSV + (
        "20260101 170500;1.174140;1.174490;1.174140;1.174480;0\n"  # duplicate time
        "20260101 170600;1.174490;1.174000;1.174600;1.174500;0\n"  # high below low
        "20260101 170700;0;0;0;0;0\n"                              # non-positive price
    )
    cleaned, dropped = clean_m1(parse_m1_csv(text))
    assert len(cleaned) == 2
    assert dropped == 3
