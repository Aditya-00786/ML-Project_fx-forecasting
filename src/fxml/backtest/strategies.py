"""Trading strategies built from the forecasts, with transaction costs.

Timing: a position chosen at the close of day t, using only what is known then, is held until
the close of the next trading day. It earns that day's return plus the interest differential
for the calendar days held (carry: holding a currency pays its interest rate, borrowing the
other costs its rate). Changing the position costs half the pair's bid-ask spread per unit
traded.

A strategy is a signal (+1 long the pair, -1 short, 0 flat) times a size:

- ``unit``: one unit of notional;
- ``vt_<model>``: volatility targeting, sized so the position's forecast volatility is
  ``TARGET_VOL`` a year, using that model's variance forecast, with leverage capped at
  ``MAX_LEVERAGE``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_VOL = 0.10  # annualised volatility a volatility-targeted position aims for
MAX_LEVERAGE = 3.0
MOMENTUM_FEATURE = "mom_252"  # 12-month return: classic time series momentum
CONFIDENCE_WINDOW = 252  # days of past forecasts that set the "confident" threshold
DIRECTION_MODELS = ("logistic_l2", "random_forest", "lightgbm", "mlp")
SIZING_MODELS = {"naive": "baseline_naive", "ewma": "baseline_ewma", "har": "baseline_har",
                 "lightgbm": "lightgbm"}


def pair_data(
    frame: pd.DataFrame, close: pd.Series, predictions: pd.DataFrame, spread: float
) -> pd.DataFrame:
    """Everything one pair's strategies need, on the days with forecasts.

    ``frame`` is the pair's modelling frame, ``close`` its daily closes, ``predictions`` its rows
    of the comparison forecasts (``reports/predictions/ml.parquet``) and ``spread`` the full
    bid-ask spread in price units.
    """
    probs = (predictions[predictions["task"] == "direction"]
             .pivot(index="date", columns="model", values="forecast").add_prefix("p_"))
    variances = (predictions[predictions["task"] == "volatility"]
                 .pivot(index="date", columns="model", values="forecast").add_prefix("var_"))
    dates = probs.index.intersection(frame.index)
    rows = frame.loc[dates]
    days_held = (rows["target_date"] - pd.Series(dates, index=dates)).dt.days
    data = pd.DataFrame({
        "ret": rows["y_ret"],
        "carry": rows["rate_diff"] / 100 * days_held / 365,
        "half_spread": spread / 2 / close.reindex(dates),
        "rate_diff": rows["rate_diff"],
        "momentum": rows[MOMENTUM_FEATURE],
        "regime": rows["regime"],
    }, index=dates)
    return data.join(probs).join(variances)


def confident(prob: pd.Series, window: int = CONFIDENCE_WINDOW) -> pd.Series:
    """Up/down calls only on days the forecast is further from 50% than its median distance
    over the previous ``window`` days of forecasts; flat otherwise."""
    distance = (prob - 0.5).abs()
    threshold = distance.shift(1).rolling(window, min_periods=21).median().fillna(0.0)
    return np.sign(prob - 0.5).where(distance > threshold, 0.0)


def signals(data: pd.DataFrame) -> pd.DataFrame:
    """Every strategy's signal for each day: +1 long, -1 short, 0 flat."""
    out = {
        "flat": pd.Series(0.0, index=data.index),
        "long": pd.Series(1.0, index=data.index),
        "carry": np.sign(data["rate_diff"]),
        "momentum": np.sign(data["momentum"]),
    }
    for model in DIRECTION_MODELS:
        prob = data[f"p_{model}"]
        out[model] = np.sign(prob - 0.5)
        out[f"{model}_confident"] = confident(prob)
    return pd.DataFrame(out).fillna(0.0)


def sizes(data: pd.DataFrame) -> pd.DataFrame:
    """Position size for each sizing rule and day."""
    out = {"unit": pd.Series(1.0, index=data.index)}
    for name, model in SIZING_MODELS.items():
        forecast_vol = np.sqrt(data[f"var_{model}"] * 252)
        out[f"vt_{name}"] = (TARGET_VOL / forecast_vol).clip(upper=MAX_LEVERAGE)
    return pd.DataFrame(out)


def strategy_returns(
    data: pd.DataFrame, signal: pd.Series, size: pd.Series, cost_multiplier: float = 1.0
) -> pd.DataFrame:
    """Daily position, gross return, trading cost and net return of one strategy."""
    position = (signal * size).fillna(0.0)
    turnover = (position - position.shift(1, fill_value=0.0)).abs()
    gross = position * (data["ret"] + data["carry"])
    cost = turnover * data["half_spread"] * cost_multiplier
    return pd.DataFrame({"position": position, "turnover": turnover, "gross": gross,
                         "cost": cost, "net": gross - cost})


def run_pair(data: pd.DataFrame, cost_multipliers=(1.0, 2.0)) -> pd.DataFrame:
    """Every signal with every sizing rule (flat only once), in long form."""
    frames = []
    signal_table, size_table = signals(data), sizes(data)
    for signal_name, signal in signal_table.items():
        for size_name, size in size_table.items():
            if signal_name == "flat" and size_name != "unit":
                continue
            result = strategy_returns(data, signal, size)
            for multiplier in cost_multipliers[1:]:
                stressed = strategy_returns(data, signal, size, multiplier)
                result[f"net_{multiplier:g}x"] = stressed["net"]
            frames.append(result.assign(signal=signal_name, sizing=size_name)
                          .rename_axis("date").reset_index())
    return pd.concat(frames, ignore_index=True)
