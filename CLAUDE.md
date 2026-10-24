# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Crypto volatility regime detection using Gaussian Hidden Markov Models. Binance OHLCV klines (BTC/ETH/DOGE/LTC at 5m/15m/30m, split by quarter of 2025) are turned into feature sets, fit with `hmmlearn.GaussianHMM`, and compared across four feature-set variants (Models A–D) using log-likelihood / AIC / BIC. Hidden states are mapped to Low, Medium, and High volatility by sorting states on mean rolling volatility. All four models use absolute log return and high-low range. B adds volume, C adds rolling volatility, and D adds both. Rolling volatility is computed from signed log returns.

The research findings below used earlier models with signed log return as an HMM input. Recalculate model comparisons before applying those figures to the current absolute-return models.

**The thesis focus is the volume feature.** Volume enters Models B/D as **raw volume** in both implementations — deliberately. Deseasonalized/log/z-scored variants were tested and score *worse* on likelihood metrics — do not "clean up" the volume feature expecting AIC to improve.

Where Model B actually beats Model A (full covariance, Oct-10 window excluded, verified 17 Aug 2026): **Q1 2025, 9 of 12 coin/timeframe cells — 6 of 6 on BTC/ETH, average +7.90% AIC.** Q2 is partial (4/12), **Q3 is 0/12, and Q4 is 1/12**. Q4 previously looked strong (7/12) only because the 2025-10-10 crash bar was still in the sample; that was a collapsed hidden state, not a regime result. Do not restore the "B wins in Q1/Q4" framing.

The signal is **not** mainly the heavy right tail. Permutation controls (refit B with volume fully permuted / block-permuted) split volume's likelihood contribution into 28.5% marginal shape, 8.6% short-run persistence, **63.0% genuine regime alignment**. The mechanism is volume's *within-state* correlation with `high_low_range` (0.25–0.76, highest in Q1) — which is exactly why `covariance_type` must stay `"full"`. Metric-independent corroboration is in `volume_analysis.py` (volume surprise is strongly positive in the high-vol regime in 12/12 BTC cells, with regimes fitted without volume).

All active code lives in `src/HMM/` (Python). The Next.js/TypeScript app tracked in git (`src/pages`, `src/components`, `package.json`, etc.) is a **deleted earlier iteration** — the deletions are uncommitted, and the Node scaffolding at the repo root is leftover. Do not resurrect or build on it unless asked.

## Commands

`main.py`, `train_2025.py`, `update_model_guide.py`, and `server.py` resolve data relative to their source files and can run from the repository root or `src/HMM/`. The fetcher still resolves its output relative to the CWD, so run `data_fetcher.py` from `src/HMM/`.

```bash
cd src/HMM
pip install flask flask-cors hmmlearn scikit-learn pandas numpy requests

python data_fetcher.py            # fetch BTC klines from Binance
python data_fetcher.py eth doge   # any of: btc, eth, doge

python main.py                    # batch run: all four coins × timeframes × quarters × Models A–D
python train_2025.py --force      # refresh all 48 annual saved models
python update_model_guide.py      # update complete tables using saved models and quarterly results
python server.py                  # Flask dashboard on http://localhost:5050
```

Forecast comparison tests use Python unittest and Node's built-in test runner:

```bash
python -B -m unittest discover -s src/HMM/tests -p "test_*.py" -v
node --test src/HMM/tests/test_dashboard_forecasts.cjs
```

`forecast_comparison.py` computes base-2 JSD/KL and exact overlap, aligns hidden-state probabilities to Low/Medium/High, and performs forward filtering. Preserve exact zero probabilities and explicit infinite KL. Live inference carries a model-signed closed-candle filter context; do not reintroduce smoothing or reset its state on every rolling 300-bar update. The 2025 in-sample quarterly fits must not be presented as historical past-only forecast comparisons. Metric calculations do not require retraining.

## Architecture

The HMM pipeline exists **twice** — keep this in mind when changing features or model settings:

1. **Batch pipeline** (`main.py`): modular — `data_loader.py` (klines JSON → typed DataFrame) → `feature_engineering.py` (`MODEL_BUILDERS` dict, Models A–D) → `hmm_model.py` (train/predict/metrics/`label_regimes`) → `export_to_json.py` (per-run JSON + `hmm_results_summary.json` into `data/processed/`).
2. **Dashboard backend** (`server.py`): self-contained Flask app that reimplements loading, features, and metrics inline (does not import the modules above). Computes on the fly per request; serves `dashboard.html` (single-file Chart.js UI) and JSON APIs: `/api/coins`, `/api/run?coin=&tf=&states=`, `/api/files`, `/api/debug`.

The two **must** stay in sync on three settings, all aligned as of 17 Aug 2026:

- **`covariance_type="full"`, `n_iter=500`** (`hmm_model.train_hmm` and `server.run_hmm`). `"diag"` is not a neutral alternative: it forces volume to be conditionally independent of the other features given the state, which drops Model B from winning 20 of 48 cells to 0 of 48. Anyone switching this back has silently reversed the thesis's central result.
- **`EXCLUDED_WINDOWS`** — the 2025-10-10 20:00–04:00 UTC liquidation cascade, dropped in `feature_engineering.EXCLUDED_WINDOWS` and `server.EXCLUDED_WINDOWS`. That bar hit −43.2% on DOGE / −37.0% on LTC (~119σ standard-scaled) and collapsed a hidden state onto a handful of bars. Excluding it is also what makes `"full"` numerically safe: with the window in, 6 fits raised exceptions and 4 more returned degenerate models; with it out, all 384 fits run clean under both covariance settings.
- Feature definitions, metrics, and regime labeling.

Historical results, including the per-quarter and per-coin tables behind the claims above, are in `findings.txt` at the repo root. Its model-comparison figures predate the change to absolute log return in all four models; recalculate those comparisons before using them as current results.

Other things that require reading multiple files to notice:

- **Data file naming contract**: `{coin}_{tf}_{quarter}_{year}.json` (e.g. `eth_15m_Q1_2025.json`) links `data_fetcher.py`, `main.py`, and `server.py`. Both pipelines do case-insensitive lookup (`find_file`).
- **Metrics free-parameter count** (`compute_metrics`, duplicated in `hmm_model.py` and `server.py`) selects the covariance term from `model.covariance_type` — keep the lookup tables in sync if covariance settings change.
- **Regime labels depend on `rolling_vol`** existing in the DataFrame regardless of whether the model variant used it as a feature — `_base_features`/`base_features` always compute it (window = `VOL_WINDOW`, 10 bars).
- **Likelihood metrics are shaped by feature distributions, not just regime information**: comparing models with different feature counts compares likelihoods over different observation spaces, and heavy-tailed or highly autocorrelated features mechanically inflate LL. Treat cross-model AIC/BIC rankings as the thesis's chosen framework, not ground truth, and use `volume_analysis.py` for distribution-independent volume evidence.
- `volume_analysis.py` is a standalone script (imports from `main.py`/`feature_engineering.py`/`hmm_model.py`, batch config only) — the dashboard does not use it.
