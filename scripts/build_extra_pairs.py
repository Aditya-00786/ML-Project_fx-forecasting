"""Build the data for the extra pairs (a stretch goal): EUR/GBP from HistData bars and USD/INR
from FRED noon rates.

Needs the core processed data (build_dataset.py, build_features.py and run_regimes.py), whose
files it leaves untouched. Writes:

- data/processed/extra_daily_bars.parquet, extra_features.parquet and extra_inputs.parquet, in
  the same layouts as the core daily_bars, features and model_inputs files;
- reports/tables/extra_data_check.csv: each extra pair's coverage and, where it has intraday
  data, how closely its noon prices follow the Federal Reserve's H.10 noon rates (implied from
  the two dollar rates for a cross).

    uv run python scripts/build_extra_pairs.py
"""

from __future__ import annotations

import pandas as pd

from fxml.config import load_config
from fxml.data.panel import policy_rate_frame
from fxml.data.quality import coverage_report, h10_check
from fxml.extra_pairs import extra_features, extra_inputs, extra_panel, implied_h10
from fxml.features.build import FEATURES


def main() -> None:
    cfg = load_config()
    processed = cfg.paths.processed
    panel = extra_panel(cfg, pd.read_parquet(processed / "daily_bars.parquet"))
    panel.to_parquet(processed / "extra_daily_bars.parquet", compression="zstd")
    macro = pd.read_parquet(processed / "macro.parquet")
    table = extra_features(cfg, panel, macro, policy_rate_frame(cfg))
    table.to_parquet(processed / "extra_features.parquet", compression="zstd")
    state = pd.read_parquet(processed / "market_state.parquet")
    inputs = extra_inputs(cfg, table, state)
    inputs.to_parquet(processed / "extra_inputs.parquet", compression="zstd")

    pairs = panel[panel.index.get_level_values("symbol").isin(cfg.extra_pairs)]
    intraday = tuple(p for p in cfg.extra_pairs if p not in cfg.daily_only)
    implied = macro.assign(**{f"h10_{pair}": implied_h10(macro, pair) for pair in intraday})
    check = coverage_report(pairs, cfg.start, cfg.end).join(h10_check(pairs, implied, intraday))
    tables = cfg.paths.reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    check.to_csv(tables / "extra_data_check.csv")

    pd.set_option("display.width", 200)
    print(f"extra_inputs.parquet: {len(inputs):,} rows for {', '.join(cfg.extra_pairs)}")
    print("\nCoverage and agreement with the H.10 noon rates")
    print(check.T.to_string())
    test = table[table.index.get_level_values("date") >= str(cfg.walk_forward.test_start)]
    missing = test[FEATURES].isna().groupby(level="pair").mean().T
    print("\nShare missing from the test period on, worst five per pair:")
    for pair in cfg.extra_pairs:
        print(f"{pair}: {missing[pair].sort_values(ascending=False).head().round(4).to_dict()}")


if __name__ == "__main__":
    main()
