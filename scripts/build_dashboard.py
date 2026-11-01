"""Build the results dashboard from the saved results.

Collects every number the dashboard shows into reports/dashboard/data.json (which the Streamlit
app reads) and writes reports/dashboard.html, a single self-contained page that opens offline:

- Verdicts: every model against its benchmark, pair by pair, with the five research questions;
- Explorer: all scorecards, horizons, regimes, backtest, Value at Risk, learning curves,
  calibration, features, extra pairs and data checks;
- Present: one large view per step for the final presentation (arrow keys).

Collecting needs the predictions in reports/predictions and data/processed/market_state.parquet.
Where those are missing, --render-only rebuilds the page from an existing data.json.

    uv run python scripts/build_dashboard.py
    uv run python scripts/build_dashboard.py --render-only
"""

from __future__ import annotations

import argparse
import json

from fxml.config import load_config
from fxml.dashboard.data import collect
from fxml.dashboard.page import render_page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true",
                        help="rebuild the page from reports/dashboard/data.json")
    args = parser.parse_args()
    cfg = load_config()
    out = cfg.paths.reports / "dashboard"
    data_file = out / "data.json"
    if args.render_only:
        data = json.loads(data_file.read_text())
    else:
        data = collect(cfg)
        out.mkdir(parents=True, exist_ok=True)
        data_file.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    page = cfg.paths.reports / "dashboard.html"
    page.write_text(render_page(data))
    for path in (page, data_file):
        print(f"{path.relative_to(cfg.paths.reports.parent)}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
