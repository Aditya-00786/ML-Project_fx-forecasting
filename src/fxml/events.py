"""Documented market stress episodes used to shade charts and to break results down.

Dates are approximate windows around each episode, chosen before looking at any model output.
"""

from __future__ import annotations

import pandas as pd

STRESS_PERIODS: dict[str, tuple[str, str]] = {
    "Global financial crisis": ("2008-09-01", "2009-03-31"),
    "Euro area debt crisis": ("2011-07-01", "2012-07-31"),
    "SNB removes franc cap": ("2015-01-15", "2015-01-31"),
    "Brexit referendum": ("2016-06-23", "2016-07-15"),
    "COVID-19 crash": ("2020-02-20", "2020-04-30"),
    "2022 tightening": ("2022-03-01", "2022-10-31"),
    "Yen carry unwind": ("2024-07-31", "2024-08-15"),
}


SHORT_NAMES: dict[str, str] = {
    "Global financial crisis": "GFC",
    "Euro area debt crisis": "Euro crisis",
    "SNB removes franc cap": "SNB",
    "Brexit referendum": "Brexit",
    "COVID-19 crash": "COVID",
    "2022 tightening": "2022 hikes",
    "Yen carry unwind": "Carry unwind",
}


def stress_label(dates: pd.DatetimeIndex) -> pd.Series:
    """Name of the stress period each date falls in, or ``"Normal"``."""
    labels = pd.Series("Normal", index=dates)
    for name, (start, end) in STRESS_PERIODS.items():
        labels[(dates >= start) & (dates <= end)] = name
    return labels
