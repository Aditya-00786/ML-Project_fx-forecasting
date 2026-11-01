# FX Regime Forecasting

Regime-aware learning of returns and volatility in G10 currency markets. This is our BITS F464 Machine Learning project.

- What we are building and why: [docs/proposal/ML_proposal.pdf](docs/proposal/ML_proposal.pdf)
- How we are building it, phase by phase: [PLAN.md](PLAN.md)

## Setup

You need [uv](https://docs.astral.sh/uv/). On macOS, install it with `brew install uv`; for other systems, see the uv docs.

```bash
git clone https://github.com/Aditya-00786/fx-regime-forecasting.git
cd fx-regime-forecasting
uv sync            # installs Python 3.12 and the exact locked packages into .venv
uv run pytest      # should pass
```

Run any script or notebook through `uv run`, for example `uv run jupyter lab`.

If you clone into an iCloud-synced folder on a Mac (Desktop or Documents with iCloud Drive), keep iCloud away from the environment and data. Otherwise it hides files inside `.venv`, which breaks imports, and tries to upload gigabytes of data:

```bash
UV_PROJECT_ENVIRONMENT=.venv.nosync uv sync && ln -s .venv.nosync .venv
mkdir data.nosync && ln -s data.nosync data
```

## Data

Data is never committed. Every dataset comes from free public sources (HistData, FRED, BIS) and is rebuilt into `data/` with three commands:

```bash
uv run python scripts/download_data.py   # about 0.7 GB over ~25 minutes; resumable, skips files already done
uv run python scripts/build_dataset.py   # daily bars, macro series and policy rates in data/processed
uv run python scripts/check_data.py      # coverage and agreement with Federal Reserve H.10 rates
```

Then build the modelling table, run the baselines (about 10 minutes) and find the regimes:

```bash
uv run python scripts/build_features.py  # features and next-day targets in data/processed/features.parquet
uv run python scripts/run_baselines.py   # walk-forward forecasts and reports/tables/baselines_*.csv
uv run python scripts/run_regimes.py     # market regimes and currency factors in data/processed/market_state.parquet
```

Then the classical machine learning models (about 3 hours of CPU time, so run them on many cores; jobs resume where they stopped):

```bash
uv run python scripts/run_ml.py --workers 8   # every model, pair and task; reports/tables/ml_*.csv
uv run python scripts/run_learning_curves.py  # Naive Bayes against logistic regression; reports/tables/ml_learning_curves.csv
uv run python scripts/run_neural.py --workers 8   # MLP and LSTM, three seeds each; joins the same ml_*.csv tables
```

Finally the backtest, Value at Risk and model explanations (a few minutes):

```bash
uv run python scripts/run_backtest.py  # strategies, costs, breakdowns and Value at Risk; reports/tables/backtest_*.csv and risk_var.csv
uv run python scripts/run_shap.py      # SHAP values of the LightGBM models; reports/tables/shap_lightgbm.csv
```

Extensions beyond the core plan (hidden Markov regimes come from `run_regimes.py`):

```bash
uv run python scripts/run_horizons.py  # 5 and 22 day volatility and direction; reports/tables/horizons_*.csv
```

Stretch goals: two extra pairs, run apart from the core seven so no core result changes, and the results dashboard:

```bash
uv run python scripts/build_extra_pairs.py            # EUR/GBP (HistData) and USD/INR (FRED noon rates); data/processed/extra_*.parquet
uv run python scripts/run_extra_pairs.py --workers 8  # their forecasts, backtest and Value at Risk; reports/tables/extra_pairs_*.csv
uv run python scripts/build_dashboard.py              # reports/dashboard.html, a self-contained page, and reports/dashboard/data.json
uv run --group dashboard streamlit run dashboard/app.py  # the same three views as a local Streamlit app
```

Open `reports/dashboard.html` in any browser, offline. It has three views: **Verdicts** (every model against its benchmark, with the five research questions), **Explorer** (every result, any pair, model or strategy) and **Present** (slides for the final presentation, moved with the arrow keys).

On a Mac, LightGBM (used from Phase 4) also needs the OpenMP runtime: `brew install libomp`.

The stored data takes about 0.25 GB. The 1-minute bars are kept as 5-minute bars, and raw ZIP files are never written to disk. The notebooks `01_data_quality` and `02_stylized_facts` document what the data looks like.

## Layout

| Folder | Contents |
|---|---|
| `configs/` | pairs, dates, walk-forward settings and cost assumptions |
| `src/fxml/` | the project package |
| `scripts/` | command line entry points |
| `notebooks/` | exploration and results notebooks |
| `dashboard/` | the results dashboard as a Streamlit app |
| `tests/` | automated checks, including look-ahead tests |
| `reports/` | generated figures and tables, and the dashboard page |
| `docs/` | proposal and final report sources |
