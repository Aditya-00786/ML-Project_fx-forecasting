"""Check the processed data and save the quality tables to reports/tables.

Fails (exit code 1) if, for any currency pair:
- the data starts more than 3 trading days after the configured start, or ends more than
  10 trading days before the configured end (HistData publishes the current month in parts);
- under 99% of trading days are present;
- noon returns correlate below 0.95 with the Federal Reserve H.10 rates in winter, in summer
  or in the weeks when US and EU daylight saving disagree. A low match in just one of these
  points to a clock error. A missing (NaN) correlation also fails.

    uv run python scripts/check_data.py
"""

from __future__ import annotations

import sys

import pandas as pd

from fxml.config import load_config
from fxml.data.quality import coverage_report, h10_check, largest_moves

MAX_START_GAP_DAYS = 3
MAX_END_GAP_DAYS = 10
MIN_COVERAGE_PCT = 99.0
MIN_H10_CORR = 0.95
H10_COLUMNS = ("return_corr_winter", "return_corr_summer", "return_corr_clock_gap")


def main() -> int:
    cfg = load_config()
    panel = pd.read_parquet(cfg.paths.processed / "daily_bars.parquet")
    macro = pd.read_parquet(cfg.paths.processed / "macro.parquet")
    tables = cfg.paths.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    coverage = coverage_report(panel, cfg.start, cfg.end)
    h10 = h10_check(panel, macro, cfg.pairs)
    moves = largest_moves(panel)
    coverage.to_csv(tables / "data_coverage.csv")
    h10.to_csv(tables / "data_h10_check.csv")
    moves.to_csv(tables / "data_largest_moves.csv", index=False)

    pd.set_option("display.width", 160)
    print("Coverage\n", coverage.round(2).to_string(), sep="")
    print("\nAgreement with Federal Reserve H.10 noon rates\n", h10.round(4).to_string(), sep="")
    print("\nLargest daily moves (check they match real events)\n",
          moves.round({"ret": 4, "abs_ret_pct": 2}).to_string(), sep="")

    pairs = coverage.loc[list(cfg.pairs)]
    problems = [f"{p}: starts {g} trading days late" for p, g in pairs["start_gap_days"].items()
                if not g <= MAX_START_GAP_DAYS]
    problems += [f"{p}: ends {g} trading days early" for p, g in pairs["end_gap_days"].items()
                 if not g <= MAX_END_GAP_DAYS]
    problems += [f"{p}: coverage {c:.2f}%" for p, c in pairs["coverage_pct"].items()
                 if not c >= MIN_COVERAGE_PCT]
    for column in H10_COLUMNS:
        problems += [f"{p}: H.10 {column} {c:.3f}" for p, c in h10[column].items()
                     if not c >= MIN_H10_CORR]

    partial = pairs["partial_days"]
    print(f"\nNote: sessions with an hour or more without data, per pair: {partial.to_dict()}")
    if problems:
        print("\nFAILED checks:\n" + "\n".join(problems))
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
