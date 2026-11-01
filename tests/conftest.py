"""Synthetic data shaped like the real processed datasets, so tests run without downloads."""

import numpy as np
import pandas as pd
import pytest

PAIRS = ("EURUSD", "USDJPY")


def _lightgbm_loads() -> bool:
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):  # installed but unloadable (no OpenMP runtime on some Macs)
        return False
    return True


HAS_LIGHTGBM = _lightgbm_loads()
needs_lightgbm = pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM cannot load here")


def synthetic_daily(dates: pd.DatetimeIndex, start_price: float, rng: np.random.Generator):
    """Daily bars with volatility clustering (a simple GARCH-like process)."""
    n = len(dates)
    variance = np.empty(n)
    ret = np.empty(n)
    variance[0] = 4e-5
    for t in range(n):
        if t:
            variance[t] = 2e-6 + 0.08 * ret[t - 1] ** 2 + 0.9 * variance[t - 1]
        ret[t] = np.sqrt(variance[t]) * rng.standard_t(6) / np.sqrt(1.5)
    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.r_[start_price, close[:-1]]
    spread = np.abs(rng.normal(0, np.sqrt(variance), n)) + 1e-4
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    rv = variance * rng.lognormal(0, 0.3, n)
    share_up = rng.uniform(0.3, 0.7, n)
    thin = (dates.month == 12) & (dates.day == 25)
    frame = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close, "noon": close,
            "n_bars": 288, "n_minutes": np.where(thin, 300, 1440), "max_gap_minutes": 5.0,
            "rv": rv, "rv_up": rv * share_up, "rv_down": rv * (1 - share_up),
            "ret": np.log(close / open_), "span_days": 1.0,
            "parkinson": np.log(high / low) ** 2 / (4 * np.log(2)),
            "garman_klass": 0.5 * np.log(high / low) ** 2,
            "thin": thin,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    frame["partial"] = frame["thin"]
    return frame


@pytest.fixture(scope="session")
def synthetic_inputs():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2009-01-01", "2012-06-29")
    symbols = {"EURUSD": 1.3, "USDJPY": 90.0, "XAUUSD": 900.0, "SPXUSD": 1000.0}
    panel = pd.concat(
        synthetic_daily(dates, price, rng).assign(symbol=symbol)
        for symbol, price in symbols.items()
    ).set_index("symbol", append=True).sort_index()
    macro = pd.DataFrame(
        {
            "vix": 20 * np.exp(np.cumsum(rng.normal(0, 0.05, len(dates)))),
            "broad_dollar": 100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(dates)))),
            "brent": 70 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates)))),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    calendar = pd.date_range(dates[0], dates[-1], freq="D", name="date")
    rates = pd.DataFrame(
        {"USD": 0.25, "EUR": 1.0, "JPY": 0.1}, index=calendar
    ).assign(EUR=lambda r: np.where(r.index >= "2011-04-13", 1.25, 1.0))
    return panel, macro, rates


@pytest.fixture(scope="session")
def synthetic_table(synthetic_inputs):
    from fxml.features.build import build_table

    panel, macro, rates = synthetic_inputs
    return build_table(panel, macro, rates, PAIRS)


# Modelling frame for the learning models (Phases 4 and 5): one pair with regimes and folds.
FIRST_REFIT = "2011-01-01"
TEST_START = "2012-01-01"
CUT = pd.Timestamp("2012-02-15")


@pytest.fixture(scope="session")
def state(synthetic_table):
    """A market state like run_regimes.py writes, with a few days missing."""
    rng = np.random.default_rng(3)
    dates = synthetic_table.index.get_level_values("date").unique()
    turbulent = pd.Series(rng.normal(0, 1, len(dates)), index=dates).rolling(20, 1).mean()
    p1 = 1 / (1 + np.exp(-4 * turbulent))
    hmm_p1 = 1 / (1 + np.exp(-6 * turbulent))  # a sharper, more persistent view of the same state
    frame = pd.DataFrame({"gmm_p0": 1 - p1, "gmm_p1": p1,
                          "hmm_p0": 1 - hmm_p1, "hmm_p1": hmm_p1}, index=dates)
    for column in ("pc1", "pc2", "pc3"):
        frame[column] = rng.normal(0, 1, len(dates))
    return frame.drop(dates[[300, 301, 500]])


@pytest.fixture(scope="session")
def frame(synthetic_table, state):
    from fxml.models.inputs import pair_frame

    return pair_frame(synthetic_table, "EURUSD", state, FIRST_REFIT)


@pytest.fixture(scope="session")
def folds(frame):
    from fxml.validation.walkforward import walk_forward

    return list(walk_forward(frame.index, frame["target_date"], TEST_START, "MS", 0.2, 5))
