# Implementation Plan

Regime-aware learning of returns and volatility in G10 currency markets (BITS F464 project).
This plan turns the approved proposal (`docs/proposal/ML_proposal.pdf`) into working code.

## Decisions

| Topic | Decision |
|---|---|
| Who writes the code | Claude implements every phase. The team reviews each phase, runs the notebooks, interprets results and writes the report. |
| Repository | Private GitHub repo `fx-regime-forecasting` under `Aditya-00786`. Teammates are added as collaborators. |
| Workflow | One branch and one pull request per phase. A teammate reviews and merges it. |
| Environment | `uv` with Python 3.12 and a committed lockfile, so `uv sync` gives everyone identical packages. |
| Data in git | Never. All data lives in `data/` (git ignored) and is rebuilt with one script. |

## Repository layout

```
fx-regime-forecasting/
├── PLAN.md                  this file
├── README.md                setup and how to run everything
├── pyproject.toml, uv.lock  pinned environment
├── configs/default.yaml     pairs, dates, paths, walk-forward settings, cost assumptions
├── data/                    (git ignored) raw/ → interim/ (5-min bars) → processed/ (daily panel)
├── src/fxml/
│   ├── data/                downloaders (HistData, FRED, BIS), cleaning, bar building, quality checks
│   ├── features/            returns, technical indicators, volatility, PCA factors, macro, feature builder
│   ├── regimes/             k-means, Ward, Gaussian mixtures, DBSCAN, regime validation
│   ├── models/              baselines, GARCH family, HAR-RV, ARIMA, scikit-learn models, neural nets
│   ├── validation/          walk-forward splitter, metrics, statistical tests
│   └── backtest/            strategies, transaction costs, volatility targeting, VaR
├── scripts/                 download_data.py, build_dataset.py, run_experiment.py
├── notebooks/               data quality, stylized facts, regimes, results
├── tests/                   look-ahead tests, walk-forward tests, metric tests
├── reports/                 figures/ and tables/ produced by the scripts
└── docs/                    proposal and final report sources
```

## Rules that hold in every phase

1. **No look-ahead.** A feature for day *t* uses only data up to the 17:00 New York close of day *t*. Targets are day *t+1*. Automated tests check this by changing future data and asserting that past features do not move.
2. **Fit inside the fold.** Scalers, PCA, GARCH, clustering and hyperparameter search are fitted on each training window only, using scikit-learn pipelines.
3. **Time order only.** The data is never shuffled. Validation is always the most recent part of the training window, followed by a 5-day embargo.
4. **Reproducible.** Fixed random seeds. Every run is driven by a config file and saves its predictions to `reports/` as Parquet, so tables and figures can be rebuilt without retraining.
5. **Baselines first.** No model is reported without the matching naive baseline and a significance test against it.

## Phases

Each phase ends with a pull request. The next phase starts only when its "done when" checks pass.

### Phase 0 — Setup
- Install `uv` and Python 3.12. Create the package, config, tests and `.gitignore`.
- Create the private GitHub repo, push the skeleton and add a GitHub Actions workflow that runs the tests on every pull request.
- **Done when:** `uv sync && uv run pytest` passes locally and on GitHub.

### Phase 1 — Data pipeline
- **Downloads** (Claude asks before downloading, with the exact file list and size):
  - HistData 1-minute bars for EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD and NZD/USD, 2006–2026.
  - The same for gold and the S&P 500 as context.
  - FRED: VIX, the broad dollar index, Brent crude and the seven H.10 daily rates. Crude oil comes from FRED because HistData's WTI files for 2024–2026 are empty.
  - BIS: daily policy rates for the eight central banks.
- **Processing:**
  - Convert HistData timestamps (fixed EST) to New York time.
  - Define each trading day as 17:00 to 17:00 New York, and drop weekend quotes.
  - Remove duplicates and bad ticks.
  - Build 5-minute bars.
  - Compute daily OHLC, close-to-close log returns, realized variance from the 5-minute returns, and the Parkinson and Garman–Klass estimators.
  - Flag thin holiday sessions.
- **Quality checks:**
  - Bar counts per day.
  - Gap report.
  - Correlation of our daily returns with the FRED H.10 rates.
- **Notebooks:** `01_data_quality.ipynb` and `02_stylized_facts.ipynb` (fat tails, volatility clustering, autocorrelation of returns compared with squared returns).
- **Done when:**
  - a daily panel exists for all 7 pairs from Jan 2006 to Oct 2026;
  - coverage is above 99% of trading days;
  - our 12:00 New York returns correlate above 0.95 with the Federal Reserve H.10 noon rates (checked by `scripts/check_data.py`).

### Phase 2 — Features, evaluation framework and baselines
- **Feature modules:** returns and momentum, technical indicators, volatility, macro and calendar.
- **Targets:** next day log realized volatility, direction and return.
- **Walk-forward splitter:**
  - expanding window;
  - first training window 2006–2011;
  - monthly test blocks from Jan 2012;
  - validation is the last 20% of each training window, followed by a 5-day embargo.
- **Metrics:** RMSE, MAE, QLIKE, Mincer–Zarnowitz R², accuracy, balanced accuracy, ROC-AUC, Brier score and out-of-sample R².
- **Statistical tests:** Diebold–Mariano (with small sample correction), Pesaran–Timmermann and Kupiec.
- **Baselines:** random walk, historical mean, EWMA, GARCH(1,1), GJR-GARCH (Student-t), HAR-RV and ARIMA.
- **Done when:** the look-ahead tests pass and a baseline results table exists for every pair and task.
- **As built:**
  - 46 features in five groups, in `src/fxml/features/build.py`. GARCH-variance and PCA features are fitted inside each fold, so they arrive with the models in Phases 3–4.
  - Holiday sessions are removed from the modelling table.
  - Training rows whose next-day label falls inside the test month are purged.
  - Commands: `scripts/build_features.py`, then `scripts/run_baselines.py`. Results are in `notebooks/03_baselines.ipynb`.

### Phase 3 — Regimes and currency factors
- **Currency factors:** PCA on the seven-pair return panel, with every pair expressed as USD per foreign currency so the "dollar" factor has a consistent sign.
- **Regime features:** short and long horizon volatility, trend strength, cross-pair correlation and VIX.
- **Clustering:** k-means, Ward, Gaussian mixtures (EM) and DBSCAN.
  - The number of regimes is chosen by silhouette, Davies–Bouldin and BIC.
  - Labels are kept stable across refits by ordering regimes by average volatility.
- **Done when:** a regime timeline figure for 2006–2026 is produced, shaded with the known crisis episodes, together with persistence statistics.
- **As built:**
  - The market state, models and selection are in `src/fxml/regimes/`; the factors are in `src/fxml/factors.py`.
  - One number of regimes is shared by every method so they can be compared. Each method and criterion votes on the 2006–2011 window, and only solutions whose smallest regime holds at least 10% of days count. The vote chose 2 (calm and turbulent). The count is fragile because the state has no sharp clusters: before the ADX warm-up fix the same vote chose 4.
  - Walk-forward labels, Gaussian mixture probabilities and factor returns come from models refitted every January from 2012. Full sample fits are used only for the timeline.
  - DBSCAN marks shock days well in hindsight but calls up to 74% of a year's days noise out of sample, so it is descriptive only.
  - For Phase 4, long momentum features are missing for most of 2006, and gold and S&P 500 features only start in March 2009 and November 2010.
  - Command: `scripts/run_regimes.py`. Results are in `notebooks/04_regimes_factors.ipynb`.

### Phase 4 — Classical machine learning
- **Volatility:** ridge, lasso, kNN, random forest and LightGBM.
- **Direction:** logistic regression (L1/L2), Gaussian Naive Bayes, kNN, decision tree, linear and RBF SVM, random forest and LightGBM.
- **Return size:** ridge and LightGBM.
- **Global vs regime-aware:** global models, plus regime-aware versions that use regime probabilities as features, plus one expert model per regime.
- **Naive Bayes vs logistic regression:** learning curves as the training window grows (RQ5).
- **Hyperparameters:** tuned once a year on the validation slice and reused for the monthly refits, which keeps compute modest.
- **Done when:** the comparison tables against the baselines are complete, with Diebold–Mariano and Pesaran–Timmermann results.
- **As built:**
  - Models are in `src/fxml/models/ml.py` and the per-pair inputs in `src/fxml/models/inputs.py`. Features are the 46 Phase 2 features plus a walk-forward GARCH(1,1) variance forecast and the three walk-forward currency factor returns.
  - Volatility models predict log realized variance; a scale factor fitted on the validation slice each year turns that into a variance.
  - Regime-aware variants of ridge, logistic regression and LightGBM use the Gaussian mixture probability of the turbulent regime as a feature (`_regime`) or one expert per regime weighted by those probabilities (`_experts`).
  - Results: LightGBM beats HAR-RV by 6% (significant on 5 of 7 pairs). Direction and returns stay at chance. Regime information does not help, and experts per regime are worse.
  - RQ5 uses two targets: direction (no signal) and a high volatility day (strong signal). On the second, Naive Bayes ranks better up to about a year of data and logistic regression wins beyond two years.
  - Commands: `scripts/run_ml.py` (about 3 hours of CPU time, run as parallel jobs), then `scripts/run_learning_curves.py`. Results are in `notebooks/05_classical_ml.ipynb`.

### Phase 5 — Neural networks
- An MLP, plus an LSTM over a window of recent days (PyTorch).
- Refitted quarterly, with early stopping on the validation slice and several seeds.
- **Done when:** the neural models appear in the same comparison tables as the rest.
- **As built:**
  - `src/fxml/models/neural.py`: an MLP (64 and 32 units) on each day's 50 inputs and an LSTM (32 units) over the last 20 trading days, for all three tasks. PyTorch runs on CPU; on Linux it installs from PyTorch's CPU-only index.
  - At each quarterly refit the network is trained with early stopping on the validation slice, then retrained on the whole training window for the chosen number of epochs. Three seeds per network; their forecasts are averaged.
  - Scoring for every model now lives in `src/fxml/models/comparison.py`, so baselines, classical models and neural networks share `reports/tables/ml_*.csv`.
  - Results: the MLP beats HAR-RV by 3.5% (significant on 3 of 7 pairs) but not LightGBM; the LSTM is no better than HAR-RV. Direction and returns stay at chance.
  - Command: `scripts/run_neural.py` (about 4 hours of CPU time, run as parallel jobs). Results are in `notebooks/06_neural_networks.ipynb`.

### Phase 6 — Backtest, risk and interpretation
- **Strategies:**
  - baselines: always long, carry, time series momentum and flat;
  - the models' direction signals;
  - those signals with volatility targeting;
  - an equal-risk portfolio across all seven pairs.
- **Costs:** pair-specific spreads, with a sensitivity run at twice the cost.
- **Risk:** daily 99% VaR from the volatility forecasts, checked with the Kupiec test.
- **Breakdowns:** by regime, stress period, pair and subperiod.
- **Interpretation:** SHAP for the tree models.
- **Done when:** the backtest tables and figures exist and answer RQ4.
- **As built:**
  - `src/fxml/backtest/`: strategies (`strategies.py`), performance and breakdowns (`performance.py`) and Value at Risk (`risk.py`); SHAP in `src/fxml/interpret.py`.
  - A position set at day t's close earns the next day's return plus the interest differential for the days held; each trade pays half the spread. Costs are also run at twice the spread.
  - Signals: flat, always long, carry, 12-month momentum and the direction forecasts of logistic regression, random forest, LightGBM and the MLP (each also in a "confident days only" version). Sizing: one unit, or volatility targeting at 10% a year with the naive, EWMA, HAR-RV or LightGBM forecast.
  - Value at Risk at 99% from every volatility model, with normal quantiles and filtered historical simulation, checked with the Kupiec and Christoffersen tests.
  - Results: no direction signal makes money after costs; carry earns a Sharpe of 0.24 (not significant). Better volatility forecasts do not improve risk targeting but do give more accurate normal Value at Risk.
  - Commands: `scripts/run_backtest.py` and `scripts/run_shap.py`. Results are in `notebooks/07_backtest_risk.ipynb`.

### Extensions (after Phase 6)
- **Calibration and Sortino:** every classifier's probabilities get an expected calibration error and a calibration slope with its standard error, and the backtest reports Sortino ratios, as the proposal promised.
- **Hidden Markov model regimes:** Baum–Welch EM in `src/fxml/regimes/hmm.py`, with filtered (forward-only) probabilities, written by `scripts/run_regimes.py` and used as model features (`*_hmm`).
- **Transformer:** a small attention model over the last 20 days in `src/fxml/models/neural.py`.
- **Longer horizons:** 5 and 22 day volatility and direction in `src/fxml/models/horizons.py` and `scripts/run_horizons.py`, with purging and an embargo of at least the horizon, and Diebold–Mariano tests that allow for overlapping forecasts.
- **Results** (`notebooks/08_extensions.ipynb`):
  - none overturns a conclusion: the direction probabilities have calibration slopes near 0;
  - HMM regimes are more persistent but add nothing to the volatility forecasts;
  - the Transformer is no better than HAR-RV;
  - at 5 and 22 days HAR-RV beats every learning model.

### Phase 7 — Report and presentation
- Final report in the same LaTeX style as the proposal, built from the saved tables and figures.
- Slides for the presentation.
- **Done when:** every research question (RQ1–RQ5) has an answer supported by a table or figure.

## Compute budget

- The daily data is small: about 5,400 rows per pair and a few dozen features.
- **Classical models:** about 178 monthly refits × 7 pairs × ~15 models gives roughly 19,000 fits, a few hours of compute that can run overnight.
- **Neural networks:** about 60 quarterly refits per pair.
- Runs are resumable: finished folds are cached, so an interrupted run continues where it stopped.

## Risks and fallbacks

| If… | then… |
|---|---|
| HistData downloads fail or have gaps | use Dukascopy (`dukascopy-node`) for the missing files, or fall back to range-based volatility from daily prices |
| Disk space runs out | keep only 5-minute bars, delete raw files after verification, drop the context instruments |
| Seven pairs are too slow | run the full model set on EUR/USD, USD/JPY and GBP/USD, and baselines plus the best models on the rest |
| Regimes are unstable | use volatility-quantile regimes and report clustering as exploratory |
| Neural nets add nothing or are too slow | smaller networks, fewer refits, report as is |
