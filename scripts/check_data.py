"""Check the processed data and save the quality tables to reports/tables.

Fails (exit code 1) if any currency pair covers under 99% of trading days, or if its noon
returns correlate below 0.95 with the Federal Reserve H.10 rates in either winter or summer.
A low match in just one season means a daylight-saving error in the timestamps.

    uv run python scripts/check_data.py
"""

from __future__ import annotations

import sys

import pandas as pd

from fxml.config import load_config
from fxml.data.quality import coverage_report, h10_check, largest_moves

MIN_COVERAGE_PCT = 99.0
MIN_H10_CORR = 0.95


def main() -> int:
    cfg = load_config()
    panel = pd.read_parquet(cfg.paths.processed / "daily_bars.parquet")
    macro = pd.read_parquet(cfg.paths.processed / "macro.parquet")
    tables = cfg.paths.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    coverage = coverage_report(panel)
    h10 = h10_check(panel, macro, cfg.pairs)
    moves = largest_moves(panel)
    coverage.to_csv(tables / "data_coverage.csv")
    h10.to_csv(tables / "data_h10_check.csv")
    moves.to_csv(tables / "data_largest_moves.csv", index=False)

    pd.set_option("display.width", 140)
    print("Coverage\n", coverage.round(2).to_string(), sep="")
    print("\nAgreement with Federal Reserve H.10 noon rates\n", h10.round(4).to_string(), sep="")
    print("\nLargest daily moves (check they match real events)\n",
          moves.round({"ret": 4, "abs_ret_pct": 2}).to_string(), sep="")

    pair_coverage = coverage.loc[list(cfg.pairs), "coverage_pct"]
    problems = [f"{p}: coverage {c:.2f}%" for p, c in pair_coverage.items() if c < MIN_COVERAGE_PCT]
    for season in ("return_corr_winter", "return_corr_summer"):
        problems += [f"{p}: H.10 {season} {c:.3f}" for p, c in h10[season].items()
                     if c < MIN_H10_CORR]
    if problems:
        print("\nFAILED checks:\n" + "\n".join(problems))
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
