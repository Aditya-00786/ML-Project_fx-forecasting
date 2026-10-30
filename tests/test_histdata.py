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


def test_timestamps_are_new_york_local_time_with_daylight_saving():
    winter = parse_m1_csv(SAMPLE_CSV)
    assert winter.index[0] == pd.Timestamp("2026-01-01 22:04", tz="UTC")  # EST = UTC-5
    summer = parse_m1_csv("20260706 120000;1.1;1.2;1.0;1.15;0\n")
    assert summer.index[0] == pd.Timestamp("2026-07-06 16:00", tz="UTC")  # EDT = UTC-4
    assert list(winter.columns) == ["open", "high", "low", "close"]


def test_clock_times_skipped_by_daylight_saving_are_dropped():
    bars = parse_m1_csv("20260308 023000;1.1;1.2;1.0;1.15;0\n20260308 170000;1.1;1.2;1.0;1.15;0\n")
    assert len(bars) == 1


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
