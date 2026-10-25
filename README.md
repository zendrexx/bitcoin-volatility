# bitcoin-volatility
Bitcoin volatility assessment using Hidden Markov Models (HMM) to identify market regimes and analyze price behavior. Applies statistical methods to classify volatility states and visualize trends for better risk understanding.

For a plain-language explanation of Models A–D and the revised research objectives, SOP questions, and evaluation requirements, see [MODEL_GUIDE.md](MODEL_GUIDE.md).

For the complete workflow organized into thesis parts, from data collection through training, comparison, and dashboard presentation, see [THESIS_SYSTEM_SUMMARY.md](THESIS_SYSTEM_SUMMARY.md).

## Live dashboard

Run `python src/HMM/server.py` and open `http://localhost:5050`. The dashboard starts in **2025 analysis** for BTC/USDT at 15m. Select BTC, ETH, DOGE, or LTC and a 5m, 15m, or 30m candle interval. Changing the coin or interval refreshes the analysis. Choose **Live market (optional)** to fetch current prices and start streaming. All four price charts use the same closing prices and dates; only the volatility regime colors vary by model. A live open candle is provisional.

The live models are fitted once using the repository's 2025 quarter files, combined into a year for each coin and timeframe. Each saved bundle contains the scaler, HMM, feature list, hidden-state-to-volatility mapping, and each state's mean 2025 volatility. The dashboard trains a missing or outdated bundle on first use and stores it under `src/HMM/data/models/2025/`, which Git ignores. Later live updates load those bundles and do not refit them. The existing 2025 liquidation-event exclusion in `server.py` also applies to these training features.

Each live model shows the probability of Low, Medium, and High volatility on the **next selected candle interval**. It combines the current state's posterior probabilities with the saved HMM transition matrix. Each card also shows the 2025 average volatility for each state. The probabilities describe HMM regimes, not future prices, and can change while the current candle is open.

The current-state percentage measures the model's assignment to that hidden state; a larger number does not mean better model accuracy or a better trade. All four models use absolute log return, `|ln(close_now / close_previous)|`, and high-low range. Model B adds volume, Model C adds rolling volatility, and Model D adds both. The absolute value makes equal-sized upward and downward moves identical inputs for the return feature. Rolling volatility remains the standard deviation of ten signed log returns. To order the fitted states as Low, Medium, and High, the server averages that measure over all 2025 training candles assigned to each state. The displayed volatility is this full-training-set state average, rather than a reading from the latest ten live candles.

Both 2025 analysis and Live market show A-B, A-C, and A-D regime agreement, mean next-state overlap, mean Jensen-Shannon divergence (JSD), mean directional KL(A || other), and Top-1 agreement over the visible window. The 2025 figures describe in-sample distributions from each quarter's existing fits and show **Compared candles**. Live figures use past-only forecasts from closed candles and show **Forecast candles**. Expand **More statistics, latest forecast, and rankings** for medians, infinite KL counts, and separate rankings; live mode also provides the latest-candle comparison. Rankings use timestamps shared by all three comparison models; JSD is the primary divergence metric. These measures describe agreement with baseline A.

JSD and KL use base-2 logarithms. JSD ranges from 0 to 1; KL is unbounded and can be infinite when another model assigns zero probability to a state A considers possible. Infinite values remain in means and medians and are displayed explicitly. Forecasts use forward filtering with past observations only, the saved 2025 scalers and models, and Low/Medium/High state alignment. Closed-candle filter context is carried through streaming updates; provisional candles are excluded from window averages and shown separately in the latest comparison. The existing overlap formula equals the requested probability-mass intersection and retains its meaning.

The 2025 quarterly fits use their entire quarters for training. Their model-implied next-state distributions are compared descriptively, with an explicit **in-sample** label; these are not forecasts made with parameters available before each historical candle. The comparison reuses the four models already fitted by each quarterly analysis request and does not train a second set. Live past-only forecasts retain their separate training-cutoff safeguards. The 2025 analysis views remain Regime Comparison and Log-Likelihood. See [MODEL_GUIDE.md](MODEL_GUIDE.md#next-state-probability-comparison) for thesis interpretation and limitations.

## Comparison tests

```bash
python -B -m unittest discover -s src/HMM/tests -p "test_*.py" -v
node --test src/HMM/tests/test_dashboard_forecasts.cjs
```

The tests cover divergence identities and bounds, asymmetric/infinite KL, exact timestamp and state alignment, causal historical prefixes, streaming context, in-sample distribution comparisons and fit reuse, past-only forecast safeguards, the default 2025 view, window statistics, rankings, and frontend/backend consistency.

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
