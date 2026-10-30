"""Download the raw data.

HistData 1-minute bars are parsed in memory and stored straight away as intraday bars of
``bars.minutes`` minutes, one Parquet file per symbol and period. FRED series and the BIS
policy rate file are small and kept as downloaded. Re-running skips files already done.

    uv run python scripts/download_data.py                    # everything
    uv run python scripts/download_data.py --sources fred bis
    uv run python scripts/download_data.py --symbols EURUSD
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import UTC, datetime

import requests
from tqdm import tqdm

from fxml.config import Config, load_config
from fxml.data import bis, fred, histdata
from fxml.data.bars import resample_bars
from fxml.data.storage import bars_dir, read_manifest, write_manifest


def download_histdata(cfg: Config, symbols: list[str], delay: float) -> list[str]:
    session = histdata.new_session()
    manifest = read_manifest(cfg)
    failures = []
    for symbol in symbols:
        offered = histdata.available_periods(symbol, session)
        periods = histdata.select_periods(offered, cfg.start, cfg.end)
        todo = [p for p in periods if f"{symbol}/{p.key}" not in manifest]
        print(f"{symbol}: {len(periods)} files offered in range, {len(todo)} to download")
        for period in tqdm(todo, desc=symbol, unit="file"):
            key = f"{symbol}/{period.key}"
            try:
                content = histdata.download_zip(symbol, period, session)
                m1, dropped = histdata.clean_m1(histdata.read_zip(content))
                bars = resample_bars(m1, cfg.bar_minutes)
                out_dir = bars_dir(cfg, symbol)
                out_dir.mkdir(parents=True, exist_ok=True)
                bars.to_parquet(out_dir / f"{period.key}.parquet", compression="zstd")
                manifest[key] = {
                    "rows_1min": len(m1),
                    "rows_dropped": dropped,
                    "rows_bars": len(bars),
                    "first": m1.index.min().isoformat(),
                    "last": m1.index.max().isoformat(),
                    "zip_bytes": len(content),
                    "zip_sha256": hashlib.sha256(content).hexdigest(),
                    "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                write_manifest(cfg, manifest)
            except Exception as exc:  # keep going; failures are reported at the end
                failures.append(f"{key}: {exc}")
                tqdm.write(f"FAILED {key}: {exc}")
            time.sleep(delay)
    return failures


def download_fred(cfg: Config) -> None:
    session = requests.Session()  # FRED stalls on browser-like user agents; use the default one
    for series_id in cfg.fred_series:
        path = fred.download_series(series_id, cfg.paths.raw / "fred", session)
        print(f"FRED {series_id}: saved {path.stat().st_size / 1e3:.0f} kB")


def download_bis(cfg: Config) -> None:
    path = bis.download(cfg.paths.raw / "bis", requests.Session())
    print(f"BIS policy rates: saved {path.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sources", nargs="+", default=["fred", "bis", "histdata"],
                        choices=["fred", "bis", "histdata"])
    parser.add_argument("--symbols", nargs="+", help="HistData symbols (default: all configured)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="pause in seconds between HistData downloads")
    args = parser.parse_args()
    cfg = load_config()

    if "fred" in args.sources:
        download_fred(cfg)
    if "bis" in args.sources:
        download_bis(cfg)
    failures = []
    if "histdata" in args.sources:
        symbols = args.symbols or [*cfg.pairs, *cfg.context_instruments]
        failures = download_histdata(cfg, symbols, args.delay)
    if failures:
        print(f"\n{len(failures)} files failed; re-run the same command to retry them:")
        print("\n".join(failures))
        return 1
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
