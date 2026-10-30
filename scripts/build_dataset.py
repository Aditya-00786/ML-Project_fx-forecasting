"""Build the processed daily datasets from the downloaded data.

Writes to ``data/processed``:
    daily_bars.parquet     one row per (date, symbol): OHLC, noon price, returns, realized variance
    macro.parquet          VIX, broad dollar index and H.10 reference rates
    policy_rates.parquet   central bank policy rate per currency, every calendar day

    uv run python scripts/build_dataset.py
"""

from __future__ import annotations

from fxml.config import load_config
from fxml.data.panel import daily_panel, macro_frame, policy_rate_frame


def main() -> None:
    cfg = load_config()
    out = cfg.paths.processed
    out.mkdir(parents=True, exist_ok=True)

    panel = daily_panel(cfg)
    clipped = {s: n for s, n in panel.attrs.pop("clipped_wicks").items() if n}
    panel.to_parquet(out / "daily_bars.parquet", compression="zstd")
    print(f"bad-tick wicks removed: {clipped or 'none'}")
    days = panel.reset_index().groupby("symbol")["date"].agg(["min", "max", "count"])
    print("daily_bars.parquet")
    print(days.to_string())

    macro = macro_frame(cfg)
    macro.to_parquet(out / "macro.parquet", compression="zstd")
    print(f"\nmacro.parquet: {len(macro)} rows, columns {list(macro.columns)}")

    rates = policy_rate_frame(cfg)
    rates.to_parquet(out / "policy_rates.parquet", compression="zstd")
    print(f"policy_rates.parquet: {len(rates)} days, missing values per currency:")
    print(rates.isna().sum().to_string())


if __name__ == "__main__":
    main()
