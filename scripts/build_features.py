"""Build the modelling table (features and next-day targets) from the processed daily data.

Writes data/processed/features.parquet, indexed by (date, pair).

    uv run python scripts/build_features.py
"""

from __future__ import annotations

import pandas as pd

from fxml.config import load_config
from fxml.features.build import FEATURE_GROUPS, FEATURES, build_table


def main() -> None:
    cfg = load_config()
    processed = cfg.paths.processed
    table = build_table(
        pd.read_parquet(processed / "daily_bars.parquet"),
        pd.read_parquet(processed / "macro.parquet"),
        pd.read_parquet(processed / "policy_rates.parquet"),
        cfg.pairs,
    )
    table.to_parquet(processed / "features.parquet", compression="zstd")

    print(f"features.parquet: {len(table):,} rows, {len(FEATURES)} features in "
          f"{len(FEATURE_GROUPS)} groups")
    test = table[table.index.get_level_values("date") >= str(cfg.walk_forward.test_start)]
    missing = test[FEATURES].isna().mean().sort_values(ascending=False)
    print("Share missing from the test period on, worst five:")
    print(missing.head().round(4).to_string())


if __name__ == "__main__":
    main()
