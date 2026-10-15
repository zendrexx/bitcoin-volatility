# bitcoin-volatility
Bitcoin volatility assessment using Hidden Markov Models (HMM) to identify market regimes and analyze price behavior. Applies statistical methods to classify volatility states and visualize trends for better risk understanding.

For a plain-language explanation of Models A–D and the revised research objectives, SOP questions, and evaluation requirements, see [MODEL_GUIDE.md](MODEL_GUIDE.md).

## Live dashboard

Run `python src/HMM/server.py` and open `http://localhost:5050`. The dashboard starts in **Live market** mode for BTC/USDT at 15m. Select BTC, ETH, DOGE, or LTC and a 5m, 15m, or 30m candle interval. All four price charts use the same Binance Spot closing prices and dates; only the volatility regime colors vary by model. The open candle is provisional. Switch to **2025 analysis** for the existing quarterly comparison and summary charts.

The live models are fitted once using the repository's 2025 quarter files, combined into a year for each coin and timeframe. Each saved bundle contains the scaler, HMM, feature list, hidden-state-to-volatility mapping, and each state's mean 2025 volatility. The dashboard trains a missing or outdated bundle on first use and stores it under `src/HMM/data/models/2025/`, which Git ignores. Later live updates load those bundles and do not refit them. The existing 2025 liquidation-event exclusion in `server.py` also applies to these training features.

Each live model shows the probability of Low, Medium, and High volatility on the **next selected candle interval**. It combines the current state's posterior probabilities with the saved HMM transition matrix. Each card also shows the 2025 average volatility for each state. The probabilities describe HMM regimes, not future prices, and can change while the current candle is open.

The current-state percentage measures the model's assignment to that hidden state; a larger number does not mean better model accuracy or a better trade. All four models use absolute log return, `|ln(close_now / close_previous)|`, and high-low range. Model B adds volume, Model C adds rolling volatility, and Model D adds both. The absolute value makes equal-sized upward and downward moves identical inputs for the return feature. Rolling volatility remains the standard deviation of ten signed log returns. To order the fitted states as Low, Medium, and High, the server averages that measure over all 2025 training candles assigned to each state. The displayed volatility is this full-training-set state average, rather than a reading from the latest ten live candles.

The 2025 analysis comparison panel reports A–B and A–C regime agreement for the visible time window. It also reports whether each pair agrees specifically on High versus non-High volatility. The available views are Regime Comparison and Log-Likelihood. Live market omits the agreement and explanatory text blocks.

To prepare models ahead of time, run `python src/HMM/train_2025.py --coin all --tf all` (or choose one coin and timeframe). Add `--force` to replace saved bundles after changing the training data or model code. This takes several minutes per selection. Live mode requires network access to Binance Spot's public market-data REST and WebSocket endpoints; it shows an error when the feed is unavailable.

## Complete 2025 comparison

Run these commands from the repository root:

```bash
python -B src/HMM/train_2025.py --coin all --tf all --force
python -B src/HMM/main.py --coin all --tf all
python -B src/HMM/update_model_guide.py
```

These refresh 48 annual saved models, fit all 192 quarterly combinations, and update the guide's tables for BTC, ETH, DOGE, and LTC at 5m, 15m, and 30m. Both training commands default to all coins and intervals. The batch accepts `--coin`, `--tf`, and `--quarter` for smaller selections and preserves unselected summary entries.

Quarterly candle files and state statistics are in `src/HMM/data/processed/`. BTC keeps filenames such as `hmm_15m_Q1_ModelA.json`; other coins use prefixes such as `hmm_eth_15m_Q1_ModelA.json`. The combined summary identifies each fit and includes explicit errors for failed fits. Training uses the existing 2025 raw dataset.
