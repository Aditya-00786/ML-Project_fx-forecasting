"""Backtest trading strategies built from the forecasts and check Value at Risk (Phase 6).

Needs the forecasts of every model (reports/predictions/ml.parquet, written by run_ml.py and
run_neural.py), the model inputs and the daily bars. Writes:

- reports/predictions/backtest.parquet: daily returns of every strategy, per pair and for the
  equal-risk portfolio;
- reports/tables/backtest_strategies.csv: performance per strategy, pair and portfolio, at
  normal and doubled trading costs;
- reports/tables/backtest_breakdown.csv: portfolio performance by regime, stress episode and
  subperiod;
- reports/tables/backtest_risk_targeting.csv: how closely each volatility forecast keeps a
  position at its risk target;
- reports/tables/risk_var.csv: 99% Value at Risk exceedances with Kupiec and Christoffersen
  tests, per volatility model, pair, method and side.

    uv run python scripts/run_backtest.py
"""

from __future__ import annotations

import pandas as pd

from fxml.backtest.performance import breakdown, portfolio, risk_targeting, strategy_table
from fxml.backtest.risk import backtest_var, exceedances, value_at_risk
from fxml.backtest.strategies import TARGET_VOL, pair_data, run_pair
from fxml.config import load_config
from fxml.events import stress_label
from fxml.features.build import trading_days

SUBPERIODS = {"2012–2016": ("2012", "2016"), "2017–2021": ("2017", "2021"),
              "2022–2026": ("2022", "2026")}


def subperiod(dates: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series(index=dates, dtype=object)
    for name, (start, end) in SUBPERIODS.items():
        out[(dates.year >= int(start)) & (dates.year <= int(end))] = name
    return out


def main() -> None:
    cfg = load_config()
    reports = cfg.paths.reports
    inputs = pd.read_parquet(cfg.paths.processed / "model_inputs.parquet")
    predictions = pd.read_parquet(reports / "predictions" / "ml.parquet")
    bars = pd.read_parquet(cfg.paths.processed / "daily_bars.parquet")
    state = pd.read_parquet(cfg.paths.processed / "market_state.parquet")

    results, var_rows = [], []
    for pair in cfg.pairs:
        frame = inputs.xs(pair, level="pair")
        close = trading_days(bars.xs(pair, level="symbol"))["close"]
        data = pair_data(frame, close, predictions[predictions["pair"] == pair],
                         cfg.costs.spread_in_price(pair))
        results.append(run_pair(data, (1.0, cfg.costs.stress_multiplier)).assign(pair=pair))
        for column in [c for c in data.columns if c.startswith("var_")]:
            hits = exceedances(value_at_risk(data[column], data["ret"]), data["ret"])
            common = hits.notna().all(axis=1)  # both methods on the same days
            for name in hits.columns:
                method, side = name.rsplit("_", 1)
                var_rows.append({"model": column.removeprefix("var_"), "pair": pair,
                                 "method": method, "side": side}
                                | backtest_var(hits.loc[common, name]))
        print(f"{pair}: {len(data):,} days", flush=True)

    per_pair = pd.concat(results, ignore_index=True)
    equal_risk = portfolio(per_pair).assign(pair="portfolio")
    daily = pd.concat([per_pair, equal_risk], ignore_index=True)
    daily.to_parquet(reports / "predictions" / "backtest.parquet", compression="zstd")

    tables = reports / "tables"
    strategies = strategy_table(daily, ["pair"])
    strategies.to_csv(tables / "backtest_strategies.csv", index=False)

    dates = pd.DatetimeIndex(sorted(equal_risk["date"].unique()))
    regimes = state["regime_gmm"].reindex(dates)  # walk-forward labels, known at each close
    top = regimes.max()
    regime_names = regimes.map(lambda r: None if pd.isna(r) else "calm" if r == 0
                               else "turbulent" if r == top else f"regime {int(r)}")
    dimensions = {
        "regime": regime_names,
        "stress": stress_label(dates),
        "subperiod": subperiod(dates),
    }
    parts = [breakdown(equal_risk, buckets, "bucket").assign(dimension=name)
             for name, buckets in dimensions.items()]
    first = ["dimension", "bucket", "signal", "sizing"]
    pieces = pd.concat(parts, ignore_index=True)
    pieces[first + [c for c in pieces.columns if c not in first]].to_csv(
        tables / "backtest_breakdown.csv", index=False)

    risk_targeting(per_pair, TARGET_VOL).to_csv(tables / "backtest_risk_targeting.csv",
                                                 index=False)
    var = pd.DataFrame(var_rows)
    var.to_csv(tables / "risk_var.csv", index=False)

    pd.set_option("display.width", 220)
    portfolio_rows = strategies[strategies["pair"] == "portfolio"].set_index(["signal", "sizing"])
    print("\nequal-risk portfolio, 2012–2026, net of costs")
    print(portfolio_rows[["ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover",
                          "cost_drag", "t_stat", "p_value", "sharpe_2x_costs"]].round(3)
          .to_string())
    print("\nValue at Risk: average exceedance rate (target 1%) and pairs passing Kupiec")
    var["kupiec_ok"] = var["kupiec_p"] >= 0.05
    print(var.groupby(["model", "method"]).agg(rate=("rate", "mean"),
                                                kupiec_ok=("kupiec_ok", "sum"))
          .unstack("method").round(4).to_string())


if __name__ == "__main__":
    main()
