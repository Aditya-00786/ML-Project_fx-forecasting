"""Assemble the processed daily datasets used by every later phase."""

from __future__ import annotations

import pandas as pd

from fxml.config import Config
from fxml.data import bis, fred
from fxml.data.bars import add_daily_measures, clip_bad_wicks, daily_bars
from fxml.data.storage import load_bars


def daily_panel(cfg: Config, symbols: list[str] | None = None) -> pd.DataFrame:
    """Daily bars for every symbol in long format, indexed by (date, symbol).

    ``panel.attrs["clipped_wicks"]`` records how many intraday bars had a bad tick removed.
    """
    symbols = symbols or [*cfg.pairs, *cfg.context_instruments]
    frames, clipped = [], {}
    for symbol in symbols:
        bars, clipped[symbol] = clip_bad_wicks(load_bars(cfg, symbol))
        daily = daily_bars(bars, cfg.timezone, cfg.day_close_hour)
        daily = daily.loc[str(cfg.start) : str(cfg.end)]
        frames.append(add_daily_measures(daily).assign(symbol=symbol))
    panel = pd.concat(frames).set_index("symbol", append=True).sort_index()
    panel.attrs["clipped_wicks"] = clipped
    return panel


def macro_frame(cfg: Config) -> pd.DataFrame:
    """VIX, broad dollar index and H.10 reference rates, one row per FRED date."""
    frame = fred.load_series(cfg.paths.raw / "fred", cfg.fred_series)
    return frame.loc[str(cfg.start) : str(cfg.end)]


def policy_rate_frame(cfg: Config) -> pd.DataFrame:
    """Policy rate (percent per year) per currency on every calendar day."""
    zip_path = cfg.paths.raw / "bis" / bis.BULK_URL.rsplit("/", 1)[-1]
    since = cfg.start.replace(year=cfg.start.year - 1)
    obs = bis.read_observations(zip_path, set(cfg.policy_rates), since)
    return bis.policy_rates(obs, cfg.policy_rates, cfg.start, cfg.end)
