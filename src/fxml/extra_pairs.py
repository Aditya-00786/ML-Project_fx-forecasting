"""Pairs studied beyond the core seven (a stretch goal): EUR/GBP, a cross with no dollar leg, and
USD/INR, a managed emerging-market currency.

They are kept apart from the core so every core result stays as it is: the market state,
regimes and currency factors still come from the seven core pairs, and the extra pairs use them
as features like any other input.

EUR/GBP has HistData 1-minute bars and goes through the core feature pipeline unchanged.

USD/INR has no free intraday data. Its prices are the Federal Reserve's noon buying rates (FRED
DEXINUS), so:

- there is no realized variance and so no volatility task. Volatility for position sizing and
  Value at Risk comes from daily returns (EWMA and GARCH);
- features that need intraday data (realized variance, ranges, ATR, stochastics, ADX) carry no
  information and are set to 0;
- each price is fixed at noon New York, five hours before the 17:00 close that dates every other
  input. A feature not computed from the pair's own noon prices (VIX, the dollar index, oil,
  gold, the S&P 500, policy rates, currency factors and regimes) is therefore taken from the
  previous trading day, so nothing is used before it is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxml.config import Config
from fxml.data import fred
from fxml.data.panel import daily_panel
from fxml.features.build import FEATURE_GROUPS, build_table
from fxml.models.inputs import FACTOR_COLUMNS, hmm_columns, pair_frame, regime_columns

INTRADAY_FEATURES = [
    "atr_pct", "stoch_k", "stoch_d", "adx_14", "logrv_d", "logrv_w", "logrv_m", "log_rv_up",
    "log_rv_down", "log_parkinson", "log_garman_klass", "vol_of_vol_22", "partial",
]
LAGGED_FEATURES = FEATURE_GROUPS["macro"]  # known only at the 17:00 close, after a noon fixing
BAR_COLUMNS = ["open", "high", "low", "rv", "rv_up", "rv_down", "parkinson", "garman_klass"]


def noon_bars(noon: pd.Series) -> pd.DataFrame:
    """A daily-only pair in the shape of the daily bars: the noon rate is the day's close and
    every intraday measure is missing."""
    noon = noon[noon > 0].dropna()
    out = pd.DataFrame({"close": noon, "noon": noon}, index=noon.index.rename("date"))
    for column in BAR_COLUMNS:
        out[column] = np.nan
    out["ret"] = np.log(noon).diff()
    out["thin"] = False
    out["partial"] = False
    return out


def extra_panel(cfg: Config, core_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily bars of the extra pairs, plus the core panel's context instruments (gold and the
    S&P 500) that the features need, indexed by (date, symbol)."""
    symbols = core_panel.index.get_level_values("symbol")
    frames = [core_panel[symbols.isin(cfg.context_instruments)]]
    intraday = [pair for pair in cfg.extra_pairs if pair not in cfg.daily_only]
    if intraday:
        frames.append(daily_panel(cfg, intraday))
    if cfg.daily_only:
        rates = fred.load_series(cfg.paths.raw / "fred",
                                 {series: pair for pair, series in cfg.daily_only.items()})
        for pair in cfg.daily_only:
            bars = noon_bars(rates[pair].loc[str(cfg.start) : str(cfg.end)])
            frames.append(bars.assign(symbol=pair).set_index("symbol", append=True))
    return pd.concat(frames).sort_index()


def extra_features(
    cfg: Config, panel: pd.DataFrame, macro: pd.DataFrame, rates: pd.DataFrame
) -> pd.DataFrame:
    """Features and targets of the extra pairs, indexed by (date, pair), in the core layout."""
    table = build_table(panel, macro, rates, cfg.extra_pairs)
    daily = table.index.get_level_values("pair").isin(list(cfg.daily_only))
    table.loc[daily, INTRADAY_FEATURES] = 0.0
    table.loc[daily, LAGGED_FEATURES] = (
        table.loc[daily, LAGGED_FEATURES].groupby(level="pair").shift(1))
    return table


def extra_inputs(cfg: Config, table: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """The extra pairs' modelling frames, indexed by (date, pair), like the core model inputs.

    The currency factors and regime probabilities come from the core market state. A
    daily-only pair takes them from the previous trading day."""
    known_at_close = [*FACTOR_COLUMNS, *regime_columns(state), *hmm_columns(state), "regime"]
    frames = []
    for pair in cfg.extra_pairs:
        frame = pair_frame(table, pair, state, cfg.walk_forward.test_start)
        if pair in cfg.daily_only:
            frame[known_at_close] = frame[known_at_close].shift(1)
        frames.append(frame.assign(pair=pair))
    return pd.concat(frames).set_index("pair", append=True)


def implied_h10(macro: pd.DataFrame, pair: str) -> pd.Series:
    """A cross pair's H.10 noon rate, implied from the two dollar rates (EUR/GBP is USD per
    EUR divided by USD per GBP)."""
    def usd_per(currency: str) -> pd.Series:
        if f"h10_{currency}USD" in macro:
            return macro[f"h10_{currency}USD"]
        return 1 / macro[f"h10_USD{currency}"]

    return usd_per(pair[:3]) / usd_per(pair[3:])
