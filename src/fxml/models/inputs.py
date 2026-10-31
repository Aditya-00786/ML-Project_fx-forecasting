"""The per-pair modelling frame used by the learning models.

It extends one pair's rows of the Phase 2 feature table with the walk-forward outputs that have
fitted parameters: the GARCH variance forecast, the currency factor returns and the market
regime probabilities. Each of these was estimated only on days before the year it describes.
"""

from __future__ import annotations

import pandas as pd

from fxml.features.build import FEATURES, on_calendar
from fxml.features.fitted import walk_forward_garch

FACTOR_COLUMNS = ["pc1", "pc2", "pc3"]
BASE_FEATURES = [*FEATURES, "garch_logvar", *FACTOR_COLUMNS]


def regime_columns(frame: pd.DataFrame) -> list[str]:
    """Regime probability columns (p0 is the calmest regime)."""
    return sorted(c for c in frame.columns if c.startswith("gmm_p"))


def regime_features(frame: pd.DataFrame) -> list[str]:
    """Probabilities given to models as features. They sum to one, so the calm one is left out."""
    return regime_columns(frame)[1:]


def pair_frame(
    table: pd.DataFrame, pair: str, state: pd.DataFrame, first_refit
) -> pd.DataFrame:
    """One pair's features, targets and regime information, indexed by date.

    ``state`` is the market state written by ``scripts/run_regimes.py``. Factor returns belong
    to their own day, so a day without them stays missing. Regime probabilities describe a
    persistent state, so the last known values are carried forward to days when some other
    pair did not trade. ``regime`` is the most likely regime.
    """
    df = table.xs(pair, level="pair").copy()
    df["garch_logvar"] = walk_forward_garch(df["ret"], first_refit)
    df[FACTOR_COLUMNS] = state[FACTOR_COLUMNS].reindex(df.index)
    probs = regime_columns(state)
    for column in probs:
        df[column] = on_calendar(state[column], df.index)
    known = df[probs].notna().all(axis=1)
    df["regime"] = df[probs].to_numpy().argmax(axis=1)
    df["regime"] = df["regime"].where(known)
    return df
