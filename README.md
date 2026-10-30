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

## Data

Data is never committed. Every dataset comes from free public sources (HistData, FRED, BIS) and is rebuilt into `data/` by the pipeline scripts added in Phase 1.

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
