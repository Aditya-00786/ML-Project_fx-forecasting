"""Features whose parameters have to be estimated, refitted once a year on earlier days only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from fxml.validation.walkforward import refit_periods

GARCH_SPEC = {"mean": "Constant", "vol": "GARCH", "p": 1, "q": 1, "dist": "t"}


def walk_forward_garch(returns: pd.Series, first_refit, freq: str = "YS") -> pd.Series:
    """Log of the GARCH(1,1) forecast of the next day's return variance, for every day.

    Parameters are estimated by maximum likelihood at ``first_refit`` and at each ``freq``
    boundary, on returns before it, then held fixed while the variance updates day by day.
    Days before ``first_refit`` use the first fit (in sample). If a refit fails to converge the
    previous parameters are kept.
    """
    pct = 100 * returns.dropna()  # percent returns keep the optimiser well scaled
    params, pieces = None, []
    for i, (start, stop) in enumerate(refit_periods(first_refit, pct.index.max(), freq)):
        result = arch_model(pct[pct.index < start], **GARCH_SPEC).fit(
            disp="off", show_warning=False
        )
        if result.convergence_flag == 0 or params is None:
            params = result.params
        through = pct[pct.index < stop]
        first = through.index[0] if i == 0 else start
        if first > through.index[-1]:
            continue
        fixed = arch_model(through, **GARCH_SPEC).fix(params)
        variance = fixed.forecast(horizon=1, start=first, reindex=False).variance["h.1"]
        pieces.append(np.log(variance / 1e4))
    return pd.concat(pieces).reindex(returns.index).rename("garch_logvar")
