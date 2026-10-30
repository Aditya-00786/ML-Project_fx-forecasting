"""The market state vector that regimes are found from: one row per trading day, shared by all
pairs, built only from trailing windows so a day's state uses nothing after its close."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxml.factors import average_correlation, usd_value_returns

STATE_COLUMNS = ["vol_short", "vol_long", "trend", "correlation", "vix"]


def market_state(
    table: pd.DataFrame, pairs: tuple[str, ...], long_window: int = 63
) -> pd.DataFrame:
    """Market-wide state on days when every pair traded.

    - ``vol_short``: log realized variance over the last 5 days, averaged over pairs
    - ``vol_long``: the same over ``long_window`` days
    - ``trend``: average ADX, the strength of trends regardless of direction
    - ``correlation``: average pairwise correlation of currency returns against the dollar
    - ``vix``: log VIX, the equity market's expected volatility
    """
    wide = table[["logrv_w", "rv", "adx_14", "log_vix"]].unstack("pair")
    complete = wide["rv"][list(pairs)].notna().all(axis=1)
    wide = wide[complete]
    rv = wide["rv"][list(pairs)]
    state = pd.DataFrame(
        {
            "vol_short": wide["logrv_w"][list(pairs)].mean(axis=1),
            "vol_long": np.log(rv.rolling(long_window).mean()).mean(axis=1),
            "trend": wide["adx_14"][list(pairs)].mean(axis=1),
            "correlation": average_correlation(
                usd_value_returns(table, pairs).loc[wide.index], long_window
            ),
            "vix": wide["log_vix"][pairs[0]],
        },
        index=wide.index,
    )
    return state[STATE_COLUMNS]
