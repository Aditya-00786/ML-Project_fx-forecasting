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
uv run python scripts/download_data.py   # about 0.6 GB over ~25 minutes; resumable, skips files already done
uv run python scripts/build_dataset.py   # daily bars, macro series and policy rates in data/processed
uv run python scripts/check_data.py      # coverage and agreement with Federal Reserve H.10 rates
```

The stored data takes about 0.25 GB. The 1-minute bars are kept as 5-minute bars, and raw ZIP files are never written to disk. The notebooks `01_data_quality` and `02_stylized_facts` document what the data looks like.

## Layout

| Folder | Contents |
|---|---|
| `configs/` | pairs, dates, walk-forward settings and cost assumptions |
| `src/fxml/` | the project package |
| `scripts/` | command line entry points |
| `notebooks/` | exploration and results notebooks |
| `tests/` | automated checks, including look-ahead tests |
| `reports/` | generated figures and tables |
| `docs/` | proposal and final report sources |
