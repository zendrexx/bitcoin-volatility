# How the four volatility models work

This guide explains the code in `src/HMM/` and answers the study's three revised research questions using the existing 2025 models and processed data. Numerical examples were checked against the local artifacts on 15 October 2026. They describe the current fits and are not a completed test of predictive performance.

For an explanation of the complete project organized into thesis parts, see [THESIS_SYSTEM_SUMMARY.md](THESIS_SYSTEM_SUMMARY.md).

Start with the [research objectives and SOP questions](#research-objectives-and-sop-questions), then read the answers on [regime characteristics](#sop-1-volatility-regime-characteristics), [feature comparison](#sop-2-feature-comparison), and [quarterly differences](#sop-3-quarterly-differences). The [complete annual profiles](#complete-annual-regime-profiles-every-coin-and-interval) and [complete quarterly comparison](#complete-quarterly-comparison-every-coin-and-interval) cover BTC, ETH, DOGE, and LTC at 5m, 15m, and 30m. Expand a coin's tables to inspect every model. The BTC 15m tables in the explanation are worked examples.

## What one model does

Each model is a **three-state Gaussian Hidden Markov Model (HMM)**. It reads a sequence of crypto price candles and assigns each candle to a hidden state. The fit learns (1) the typical feature values and relationships within each state and (2) how likely the model is to move between states from one candle to the next. All four models use **full covariance**, so a state can capture relationships between inputs, such as volume and price range.

The state numbers in a JSON file (`0`, `1`, `2`) have no fixed meaning. After fitting, the code calculates the average rolling volatility of the training candles assigned to each state. It orders those averages and names the states **Low volatility**, **Medium volatility**, and **High volatility**. These names describe the size of price movement, not whether price is rising or falling.

## Inputs to Models A–D

| Model | Inputs used to fit the HMM | What changes |
| --- | --- | --- |
| A | Absolute log return; high-low range | Uses the size of the close-to-close move without its direction. |
| B | Absolute log return; high-low range; volume | Adds traded volume to Model A's inputs. |
| C | Absolute log return; high-low range; rolling volatility | Adds recent return variation to Model A's inputs. |
| D | Absolute log return; high-low range; volume; rolling volatility | Uses both extra features. |

All inputs are standardized before fitting: the training mean is subtracted and the result is divided by the training standard deviation for that input. The HMM then learns its own state means, covariances, and transition probabilities. **Volume has no manually chosen weight.**

**Comparisons:** All four models now use the same absolute log return and high-low range inputs. A versus B examines adding volume, while C versus D examines adding volume when rolling volatility is already included. A versus C and B versus D examine adding rolling volatility. This keeps the shared feature definitions consistent. Likelihoods across different feature sets still describe different observation spaces, so a higher likelihood alone does not establish better regime detection or forecasting.

## Where the features come from

The raw data are Binance Spot OHLCV candles in `src/HMM/data/raw/`. The batch runner and dashboard resolve this path relative to their source files, so they use the same files when launched from the repository root or `src/HMM/`. A candle contains its open, high, low, close, and **volume**. The `volume` field is the amount of the **base asset** traded during that interval: for BTC/USDT it is BTC, and for ETH/USDT it is ETH. It is not USDT turnover. The data loader reads this field directly from each raw kline; Models B and D pass it to the HMM after standardization. There is no volume surprise calculation in those model inputs.

For candle `t`, the derived price features are:

- **Signed log return:** `ln(close[t] / close[t-1])`. Its sign records the direction of the close-to-close move. It is retained for descriptive statistics and for computing rolling volatility.
- **Absolute log return:** the absolute value of the signed log return. All four HMMs use this return input, so equal-magnitude positive and negative log returns have the same value.
- **High-low range:** `(high[t] - low[t]) / close[t]`. This describes the candle's price span relative to its close.
- **Rolling volatility:** the sample standard deviation of the most recent **10 signed log returns**: `RV[t] = sqrt(sum((r[i] - mean_of_10_returns)^2) / 9)`, for `i = t-9, ..., t`. This is a feature in Models C and D. It is also calculated for every model so the fitted states can be named by volatility. The denominator follows the `ddof=1` default in [pandas Rolling.std](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.Rolling.std.html).

The model is fitted separately for each coin and candle interval. Raw volume numbers from different coins are therefore not treated as directly interchangeable.

## Training, processed files, and the live dashboard

The **batch pipeline** (`main.py`) fits Models A-D for every supported coin, quarter, and 5m, 15m, or 30m interval: 192 quarterly combinations. Outputs are stored in `src/HMM/data/processed/`. BTC retains names such as `hmm_15m_Q1_ModelA.json`; the other coins have explicit prefixes, such as `hmm_eth_15m_Q1_ModelA.json`, to prevent collisions. Each processed row keeps the raw candle values, signed and absolute log returns, range, rolling volatility, state number, and volatility label. The combined `hmm_results_summary.json` identifies each fit by coin, interval, quarter, and model and includes state feature statistics, occupancy, run lengths, transitions, and fit status. The server also supports quarterly fits for all four coins.

From the repository root, run `python -B src/HMM/main.py --coin all --tf all` to rebuild all quarterly outputs. The defaults also cover every coin and interval. Use `--coin eth --tf 5m` to select a subset; the combined summary preserves results for unselected combinations. Refresh the saved annual fits with `python -B src/HMM/train_2025.py --coin all --tf all --force`, then rebuild this guide's tables with `python -B src/HMM/update_model_guide.py`.

The **live dashboard** fits each model once on the combined 2025 quarter files for the selected coin and interval, then saves the scaler and HMM under `src/HMM/data/models/2025/`. It applies those saved fits to later market candles. The current-state percentage is the HMM's probability for its selected current state. The next-state percentages combine the current state probabilities with the learned transition probabilities; they estimate regimes for the next candle interval, not the next price.

The displayed **2025 average volatility** is a training-set description of each state. For each training candle, the code calculates a 10-return standard deviation; it then averages that value across all 2025 training candles assigned to the state. It is not a reading from just the latest ten live candles, and the HMM does not directly output that volatility percentage.

Quarterly processed states and live states can differ because they come from different fits. The displayed price line comes from the same market candles for all four models; the regimes and probabilities vary by model.

All 48 annual saved models have been retrained across the four coins and three intervals using absolute log return in A-D. The quarterly batch now rebuilds all 192 combinations. BTC 15m is an explanatory example throughout the guide, rather than the limit of training or result coverage.

For the current BTC 15m saved fits, the 2025 state-average volatility values are:

| Model | Low (%) | Medium (%) | High (%) |
| --- | --- | --- | --- |
| A | 0.1094 | 0.2066 | 0.4005 |
| B | 0.1151 | 0.2070 | 0.3862 |
| C | 0.0981 | 0.1938 | 0.4064 |
| D | 0.1026 | 0.1980 | 0.4063 |

These are averages over the training candles assigned to each state, with the documented HMM event-window exclusion applied.

## Questions and answers

**Why is `volume` present in a Model A or C processed row?** It is a retained raw candle field for inspection. The table above identifies the columns actually fed into each HMM.

**Does High volatility mean the price will fall?** No. High volatility means larger or more variable movements in the training data assigned to that state. It does not specify direction.

**Does a higher current-state probability mean a better model?** No. It means that model is more certain about its state assignment for that candle. Model accuracy needs separate evaluation.

**Does using absolute log return in all four models also change the rolling-volatility formula?** The direct return input is now absolute log return in A–D. Rolling volatility continues to use the standard deviation of ten signed log returns. Taking the standard deviation of absolute returns would measure variation in the magnitudes of moves, which is a different calculation.

**How is average volatility measured over the whole year?** The year is the period being summarized; ten returns is the size of each individual measurement window. The calculation has three steps:

1. Calculate each candle's signed logarithmic return: `r[t] = ln(close[t] / close[t-1])`.
2. Calculate `RV[t]`, the sample standard deviation of ten consecutive returns. Advance by one candle and repeat, producing an overlapping sequence of volatility measurements throughout the year.
3. Average all valid measurements in the selected period: `year average RV = sum(RV[t]) / number of valid RV measurements`.

For an illustrative example, if three window measurements were 0.10%, 0.20%, and 0.60%, their average would be `(0.10 + 0.20 + 0.60) / 3 = 0.30%`. Each ten-return window spans about 150 minutes at 15m, and the averaging step summarizes the valid windows across the year. The result retains the scale of the candle returns; it is not the year's total price change or an annualized value.

**How does that differ from calculating volatility using the entire year's returns at once?** That is another statistic: take the sample standard deviation of all valid returns in the year, instead of averaging the standard deviations of short windows.

| Calculation | Meaning |
| --- | --- |
| Average of ten-return rolling standard deviations | Average short-window variability over the year. |
| Standard deviation of all annual-sample returns | Dispersion across the entire year's set of candle returns. |

These are different operations, so the values need not match. Both retain the scale of the selected candle interval; neither is annualized.

**What does the average shown for each model's Low, Medium, or High state mean?** The dashboard uses the same ten-return volatility measurements, but averages only the 2025 training candles assigned to that particular state: `state average RV = sum(RV[t] for candles assigned to the state) / number of candles assigned to the state`. The live models' training sample also applies the documented event-window exclusion. All four models start from the same measured volatility series for a given coin and interval. Their state averages can differ because the models assign different candles to each state. A new live candle changes the inferred state and probabilities; it does not update these saved 2025 state averages unless the model is retrained.

## Research objectives and SOP questions

The study examines how the same HMM structure behaves when its input features change. The questions below use the revised wording. In this implementation, **"log returns" as a model input means absolute logarithmic returns**. Signed logarithmic returns are retained to calculate rolling volatility and describe return dispersion. State this operational definition in the thesis; it affects what the models can identify.

### General objective

To analyze cryptocurrency volatility regimes using Hidden Markov Models and compare how different feature combinations affect their identification across cryptocurrencies, trading intervals, and quarterly periods in the 2025 dataset.

### Specific objectives

1. Analyze the characteristics of the low, medium, and high volatility regimes identified by Hidden Markov Models in Bitcoin, Ethereum, Dogecoin, and Litecoin across different trading intervals.
2. Compare traditional indicators (log returns and high-low price range) and additional indicators (trading volume and rolling volatility) in detecting cryptocurrency volatility regimes using Hidden Markov Models.
3. Examine the performance of different feature combinations in detecting volatility regimes across quarterly periods within the one-year dataset for Bitcoin, Ethereum, Dogecoin, and Litecoin.

### Statement-of-the-problem questions

1. What are the characteristics of the low, medium, and high volatility regimes identified by Hidden Markov Models in Bitcoin, Ethereum, Dogecoin, and Litecoin across different trading intervals?
2. What differences can be observed between traditional indicators (log returns and high-low price range) and additional indicators (trading volume and rolling volatility) in detecting cryptocurrency volatility regimes using Hidden Markov Models?
3. What differences can be observed in the performance of different feature combinations in detecting volatility regimes across quarterly periods within the one-year dataset for Bitcoin, Ethereum, Dogecoin, and Litecoin?

Here, "traditional" identifies the study's baseline price features. Models B, C, and D **add** features to that baseline; they do not replace it with a volume-only or rolling-volatility-only model. "Performance" needs an explicit definition: the current evidence concerns descriptive regime quality, while predictive performance requires evaluation on later observations that were not used for fitting.

## SOP 1: Volatility regime characteristics

**Answer:** Each regime is characterized by the return magnitudes, price ranges, trading activity, and rolling volatility of the candles assigned to it, together with how frequently it occurs and how long it persists. Low, Medium, and High describe relative movement levels within a particular coin, interval, and fitted sample. Their numerical characteristics can differ across coins, intervals, and models. SOP 1 therefore profiles each state's observations and temporal behavior; SOP 2 examines how these profiles change when features are added.

### Which characteristics answer this question?

| Characteristic | What to summarize within each state | Interpretation |
| --- | --- | --- |
| Logarithmic returns | Mean and distribution of absolute log returns; standard deviation of signed log returns. | Typical movement magnitude and return dispersion. A signed mean near zero can conceal large positive and negative moves. |
| High-low price range | Mean, median, and spread of `(high - low) / close`. | Intracandle price movement, including movement not captured by the close-to-close return. |
| Trading volume | Mean, median, and spread of base-asset volume per candle. | Trading activity associated with the regime, including for models fitted without volume. Volume is an observed characteristic, not the rule used to name the states. |
| Rolling volatility | Mean, median, and spread of the ten-return sample standard deviation. | Recent return variability associated with the regime. Its mean determines the Low/Medium/High label order. |
| State frequency | Number and percentage of retained candles assigned to the state. | How common the regime is within the selected sample. |
| Duration and transitions | Consecutive run lengths in candles and minutes, switching frequency, and transition probabilities. | How long the regime persists and how the fitted process moves between regimes. |

For example, in the BTC 15m annual Model A fit, Low / Medium / High mean absolute log returns are **0.0737% / 0.1567% / 0.3821%**, mean ranges are **0.1386% / 0.3026% / 0.7360%**, and mean volumes are **89.41 / 205.66 / 636.73 BTC per candle**. Their state shares are **41.01% / 44.90% / 14.09%**. These measured characteristics explain the states beyond their names. The [model-and-state feature table](#observed-feature-profiles-by-model-and-state) provides the corresponding values for A-D.

The number of states is set to **three before training**. SOP 1 characterizes the resulting regimes. Whether three states are optimal would require a separate model-selection study.

### How the regimes are identified

For each candle, the code constructs the selected features and standardizes them using the training sample. The HMM learns a mean feature vector and a full covariance matrix for each hidden state, together with a transition matrix. In notation, `x[t] given state k ~ Gaussian(mu[k], covariance[k])`, and the transition matrix describes the probability of moving from one hidden state to another. Full covariance is used in **A, B, C, and D**.

For historical training descriptions, `predict()` decodes a Viterbi state sequence. The code then averages rolling volatility within each decoded state and orders the states from lowest to highest. The [hmmlearn API](https://hmmlearn.readthedocs.io/en/stable/api.html#hmmlearn.hmm.GaussianHMM) documents the Gaussian model, Viterbi decoding, and state posterior probabilities used here.

| Label | Interpretation within a fitted sample | Evidence to inspect |
| --- | --- | --- |
| Low volatility | Relatively quiet price movement. | Return magnitude, return dispersion, and price range of assigned candles. |
| Medium volatility | An intermediate group between the quieter and more volatile observations. | Whether this group is sufficiently distinct from both neighboring states. |
| High volatility | Relatively large or variable price movement. | Larger movements, their distribution, and whether the state represents a recurring group rather than a few observations. |

These labels are relative. A Low state in one coin can have larger price movements than a Medium state in another. There is no shared numerical threshold such as "every candle above 0.5% is High." The model uses the joint feature patterns and the sequence of observations.

### What the current saved models show

The repository contains 48 saved annual fits: **4 coins x 3 intervals x 4 feature combinations**. Reapplying their saved scalers and HMMs to the original training rows reproduced the stored state-average rolling volatility values. Every decoded state occupied at least 10.21% of its model's annual training sample, above the code's 1% minimum. This establishes that these saved fits have three populated states; occupancy alone does not establish separation or optimality.

The following table shows **Model A**, the common baseline, for every coin and interval. Values are the mean ten-return rolling standard deviation of candles assigned to each state, multiplied by 100. They are descriptive training averages, not state thresholds, predicted prices, or annualized volatility.

| Coin | Interval | Low (%) | Medium (%) | High (%) |
| --- | --- | --- | --- | --- |
| BTC | 5m | 0.0604 | 0.1154 | 0.2418 |
| BTC | 15m | 0.1094 | 0.2066 | 0.4005 |
| BTC | 30m | 0.1590 | 0.2921 | 0.5272 |
| ETH | 5m | 0.1107 | 0.2057 | 0.3926 |
| ETH | 15m | 0.2021 | 0.3640 | 0.6400 |
| ETH | 30m | 0.2963 | 0.5176 | 0.8354 |
| DOGE | 5m | 0.1464 | 0.2613 | 0.4960 |
| DOGE | 15m | 0.2657 | 0.4584 | 0.7978 |
| DOGE | 30m | 0.4358 | 0.5874 | 0.9693 |
| LTC | 5m | 0.1438 | 0.1921 | 0.3713 |
| LTC | 15m | 0.2615 | 0.3370 | 0.6319 |
| LTC | 30m | 0.3389 | 0.6016 | 1.0237 |

For example, at 15m, Model A's High-state average is 0.4005% for BTC and 0.7978% for DOGE. These are averages over different sets of candles selected by independently fitted models. They describe those states and do not rank the models' quality. Litecoin's Low and Medium averages are closer than Bitcoin's in relative terms, which motivates examining overlap rather than assuming equally clear separation for every coin.

The ten-return window covers 50 minutes at 5m, 150 minutes at 15m, and 300 minutes at 30m. Consequently, changing interval changes both the return horizon and the physical duration of the rolling window. Larger values at a longer interval do not by themselves mean a model performs better.

**How far this answers SOP 1:** The complete annual tables below profile all 144 states from all 48 saved models using all four features, frequency, observed duration, and self-transition probabilities. The quarterly summary also stores mean, median, standard deviation, and quartiles for each feature within every successfully fitted state. Increasing Low-to-High rolling-volatility means follow the labeling rule; the other characteristics explain how the states differ. Distribution overlap and repeated-fit stability require additional analysis before claiming robust separation.

## SOP 2: Feature comparison

**Answer:** Adding trading volume or rolling volatility changes the information used to assign states, and therefore changes state membership, typical feature values, and transition behavior. The observed differences depend on the coin and interval. The current results show these differences, but they do not establish a universally superior feature combination.

### What each comparison measures

| Comparison | Feature added | Interpretation |
| --- | --- | --- |
| A to B | Trading volume | Does trading activity change the regimes obtained from return magnitude and price range? |
| A to C | Rolling volatility | Does recent return variation change the regimes obtained from the two price features? |
| C to D | Trading volume | What does volume contribute when rolling volatility is already included? |
| B to D | Rolling volatility | What does recent return variation contribute when volume is already included? |

Model A is the reference model. Agreement with A describes similarity to the baseline; A's labels are not known true labels that B, C, and D must reproduce.

**Volume in B and D** adds the amount traded during the candle. Full covariance allows the HMM to represent relationships between volume and the price features within each state. Volume can therefore change a state assignment even when two candles have similar return magnitudes. An observed association between volume and volatility does not establish that volume causes the price movement.

**Rolling volatility in C and D** adds a summary of ten signed returns. It gives the model information about recent variation beyond the current candle's absolute return and range. Because neighboring windows share nine returns, the feature itself has substantial overlap over time. Smoother regime sequences may reflect this construction; smoothness alone is not evidence of greater accuracy. Rolling volatility is derived from prices, so it is not an independent external source of information.

**Model D** incorporates both additions. Its greater flexibility can change the fit, but improvement is an empirical question. With three states and full covariance, the code counts 23 free parameters for A, 35 for B, 35 for C, and 50 for D.

### Observed feature profiles by model and state

This example uses the **BTC 15m annual saved fits**, with 34,997 training candles per model. Every row summarizes the original, unstandardized values of candles assigned to that state. Return magnitude, range, and rolling volatility are expressed as percentages; volume is BTC traded per 15-minute candle. "Share" is the fraction of training candles in that state.

| Model | State | Mean absolute log return (%) | Mean range (%) | Mean volume (BTC) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.0737 | 0.1386 | 89.41 | 0.1094 | 41.01 |
| A | Medium | 0.1567 | 0.3026 | 205.66 | 0.2066 | 44.90 |
| A | High | 0.3821 | 0.7360 | 636.73 | 0.4005 | 14.09 |
| B | Low | 0.0760 | 0.1445 | 76.77 | 0.1151 | 41.93 |
| B | Medium | 0.1567 | 0.3043 | 202.60 | 0.2070 | 43.76 |
| B | High | 0.3774 | 0.7173 | 683.84 | 0.3862 | 14.32 |
| C | Low | 0.0781 | 0.1484 | 96.09 | 0.0981 | 39.76 |
| C | Medium | 0.1517 | 0.2909 | 197.00 | 0.1938 | 42.19 |
| C | High | 0.3290 | 0.6355 | 539.79 | 0.4064 | 18.04 |
| D | Low | 0.0772 | 0.1461 | 81.49 | 0.1026 | 40.17 |
| D | Medium | 0.1531 | 0.2960 | 196.45 | 0.1980 | 43.32 |
| D | High | 0.3458 | 0.6631 | 611.26 | 0.4063 | 16.50 |

All four features are profiled for every model, including those a model did not use as inputs. For example, Model A's High-state candles have higher average volume even though A was fitted without volume. That is a descriptive relationship in the data. The baseline also separates mean absolute return and range across its three states; its interpretation is supported by more than the ordering of rolling volatility alone.

In this example, B's High state has a higher mean volume than A's, while C assigns more candles to High than A does (18.04% versus 14.09%). C's High-state mean absolute return is lower than A's even though its mean rolling volatility is slightly higher. Those are different state definitions produced by the feature combinations; neither difference by itself proves superior detection.

These empirical averages are different from the HMM's learned Gaussian means. Training estimates emission parameters using probabilistic state weights; the table groups candles by their decoded state. To inspect a learned mean in original units, use `original_mean = standardized_mean * scaler.scale_ + scaler.mean_`. The displayed dashboard state-average volatility uses the empirical calculation.

### How much the models agree across coins and intervals

Agreement below means the percentage of **identical training timestamps** receiving the same Low, Medium, or High label after label ordering. Each comparison uses annual saved models and full training sequences, with Viterbi decoding. It is a description of agreement, not a correctness score.

| Coin | Interval | A-B agreement (%) | A-C agreement (%) | A-D agreement (%) |
| --- | --- | --- | --- | --- |
| BTC | 5m | 82.83 | 82.46 | 82.73 |
| BTC | 15m | 84.18 | 80.96 | 82.26 |
| BTC | 30m | 83.80 | 77.80 | 80.54 |
| ETH | 5m | 83.59 | 79.50 | 81.14 |
| ETH | 15m | 84.32 | 78.17 | 79.91 |
| ETH | 30m | 83.71 | 73.91 | 77.80 |
| DOGE | 5m | 81.71 | 80.53 | 81.41 |
| DOGE | 15m | 82.18 | 77.89 | 80.15 |
| DOGE | 30m | 66.58 | 63.31 | 64.99 |
| LTC | 5m | 61.06 | 63.96 | 61.41 |
| LTC | 15m | 59.55 | 87.52 | 60.71 |
| LTC | 30m | 84.09 | 76.95 | 79.83 |

At BTC 15m, B is closer to A than C is. At LTC 15m, C is substantially closer to A than B is. This supports the observation that changing features affects different coins differently. It does not establish that closeness to A is desirable or that one model is consistently more effective.

Both 2025 analysis and Live market show a compact agreement table for A-B, A-C, and A-D, together with the fraction of compared candles where all four agree. Same-regime percentages use all valid candle timestamps in the visible window, independently of chart downsampling. Each percentage includes its agreeing / compared counts underneath. Different pairs can have the same denominator because they use the same candle timestamps, while their agreeing counts differ. Missing or failed assignments are excluded rather than counted as matches. Sliding, zooming, or receiving live updates refreshes the table.

The dashboard opens in **2025 analysis**, with Live market available as an optional selection. Both modes compare next-state distributions using overlap, Jensen-Shannon divergence, directional KL divergence, and Top-1 agreement. The 2025 values describe the existing quarterly fits in-sample and show a **Compared candles** count. Live values compare past-only forecasts and show **Forecast candles**; live latest-candle values are reported separately from window averages. These procedures use different samples and decoding rules from the annual Viterbi descriptions above.

### Next-state probability comparison

The reference is **Model A**, and the comparisons are A-B, A-C, and A-D. The current implemented baseline uses **absolute log return and normalized high-low range**, as listed earlier. The comparison feature does not change Model A to raw High, raw Low, and signed Log Return, and it does not refit models to obtain similarity statistics.

For each prediction origin, all probability vectors are aligned to **Low / Medium / High** using the saved training-state mapping. Let `p` be A's next-state distribution and `q` be the comparison model's distribution. The sum runs over the three regimes.

| Measure | Calculation | Interpretation |
| --- | --- | --- |
| Next-state overlap | `100 * sum(min(p[i], q[i]))` | Shared probability mass; 100% means identical distributions and 0% means disjoint distributions. Higher indicates greater agreement. |
| Jensen-Shannon divergence (primary) | With `m = (p+q)/2`, `JSD = (KL(p || m) + KL(q || m))/2`. | Base-2 JSD is symmetric and ranges from 0 to 1. Lower indicates greater similarity. The implementation reports divergence, rather than the square-root Jensen-Shannon distance. |
| Directional Kullback-Leibler divergence | `KL(A || other) = sum(p[i] * log2(p[i]/q[i]))`. | Base-2 KL is in bits, is unbounded, and gives greater weight to discrepancies in states A considers likely. A is always the first distribution. Smaller indicates greater agreement with A. |
| Top-1 next-state agreement | Percentage of aligned origins where `argmax(p) == argmax(q)`. | Whether the models select the same most likely next regime. Ties are resolved deterministically in Low, Medium, High order. |

The previous overlap expression, `100 * (1 - sum(abs(p-q))/2)`, is mathematically equal to probability-mass intersection for normalized distributions. Its meaning is preserved. The inputs are validated as finite, nonnegative vectors with positive mass, and normalized before comparison. The code applies **no epsilon clipping or probability smoothing**. Terms with `p[i] = 0` contribute zero to KL. If `p[i] > 0` and `q[i] = 0`, KL is infinite. Tiny negative divergence values caused by floating-point roundoff, within a tolerance of `1e-12`, are floored at zero; infinite values are preserved.

**Window statistics:** The main table reports mean overlap, mean JSD, mean KL, Top-1 agreement, and each pair's number of aligned comparison origins. The 2025 panel counts analyzed training-sample candles; live mode counts closed forecast origins. Column-heading tooltips explain each measure and whether higher or lower indicates more agreement with A. In live mode, expand **Latest forecast** for the latest-candle comparison. Infinite KL observations remain in the sample: one infinite observation makes the arithmetic mean infinite. JSON uses the explicit string `"Infinity"` so responses remain valid JSON; the dashboard displays an infinity symbol.

**Latest-candle comparison:** The latest overlap, JSD, KL, and Top-1 result are in a separate table with the origin candle's UTC time and open/closed status. These values always refer to the latest forecast, even if the chart is panned to older dates. An open candle's forecast can change; it is excluded from the closed-window averages.

**Causal timing:** At candle `t`, forward filtering computes `P(S_t | X_1, ..., X_t)` and multiplies it by the saved transition matrix to obtain the next-state distribution. It uses no backward smoothing. Rolling features use observations through `t`, the scaler and model remain those trained in 2025, and the state ordering is fixed from that training sample. The response records origin, information-availability, and target timestamps. A closed origin is represented by its candle-open timestamp; its feature values are available at that candle's close, before the next candle's target timestamp. Streaming updates carry the most recent closed-candle filtered state, validate that the fitted model has not changed, and recompute the provisional candle from that state. Sliding the rolling input buffer therefore does not reset the HMM's filtering history. Missing-model forecasts remain absent.

Live forecast comparisons are suppressed for origins at or before the saved model's training cutoff. The existing 2025 quarterly models were trained on their entire quarters, and no saved pre-quarter forecasting models are available. Reporting their in-sample probabilities as past-only historical forecasts would leak future information through the fitted parameters and scalers.

**Descriptive 2025 analysis:** The historical panel therefore compares model-implied next-state distributions **conditional on each full-quarter fit**, with an explicit in-sample label. For a fixed fitted model, forward filtering through candle `t` and the transition matrix produce `P(S[t+1] | X[1:t], fitted_quarter_parameters)`. These vectors reuse each model already fitted for the quarterly chart; no second set of HMMs is trained to calculate the metrics. Their overlap, JSD, KL, and Top-1 agreement describe how closely B-D resemble A under the fitted quarterly models. The count is labeled **Compared candles**, not Forecast candles. Failed fits remain gaps in the distributions as well as the state traces. The API uses the distinct `distribution_comparison` field, records `method = in_sample_quarter_fit`, and explicitly sets `is_out_of_sample` and `past_only_forecasts_available` to false. This descriptive comparison is separate from the live `forecast_comparison` data. Producing a valid prospective 2025 forecast study still requires a chronological evaluation with models trained before each evaluation period.

**Reading the comparison:** Higher mean overlap and Top-1 agreement indicate more similarity to A; lower mean JSD and mean KL indicate more similarity. These measures can favor different models because overlap measures shared mass, JSD balances both distributions, and directional KL emphasizes states A favors. Compare the candle counts and use matching timestamp coverage when interpreting differences. The dashboard presents the values and explanatory tooltips without a separate ranking table or weighted composite score. Similarity to A does not establish predictive accuracy.

For the thesis, describe these results as **agreement with the baseline's next-state distribution**. A model can agree strongly with A while both forecasts are wrong about a later market outcome. These metrics therefore cannot establish predictive accuracy, true volatility labels, or investment performance. JSD is the primary divergence measure; directional KL supports its interpretation, especially when another model assigns very low or zero probability to a state A considers likely.

The formulas and zero conventions follow the official [SciPy Jensen-Shannon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html) and [SciPy relative-entropy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html). Implementation is in [forecast_comparison.py](src/HMM/forecast_comparison.py), with tests in `src/HMM/tests/`.

## SOP 3: Quarterly differences

**Answer:** Independent quarterly fits show how the same feature combination produces different state characteristics, proportions, and switching patterns in different periods. The complete quarterly tables below apply the same descriptive measures across all four coins and all three intervals. The BTC 15m example explains how to interpret those measures. A claim of predictive superiority still requires a common target and chronological evaluation on unseen observations.

### Which fits answer the quarterly question?

The quarterly analysis fits A-D separately within Q1, Q2, Q3, and Q4. Each fit has its own scaler, parameters, and label ordering. For all four coins and three intervals, the full study contains **192 quarterly fits**: `4 coins x 3 intervals x 4 quarters x 4 models`.

The current `main.py` processes all 192 combinations and retains an explicit status for each fit. Successful fits produce candle-level files and state profiles; failures retain an error in the summary. The 48 saved annual models used for live inference are a separate set of fits and must not be counted as quarterly fits.

### Observed quarterly example: BTC at 15m

These results were calculated directly from the existing `hmm_15m_Q*_Model*.json` files. All four models use matching timestamps and close prices within each quarter. Valid row counts are Q1: 8,630; Q2: 8,726; Q3: 8,822; Q4: 8,789. The quarterly samples lose ten warmup candles per quarter; Q4 also applies the event exclusion described below.

In the next table, the three rolling-volatility values are **Low / Medium / High**, in percent. "Switch rate" is the percentage of adjacent, consecutive 15-minute candle pairs whose regime changes. It excludes the jump across the removed event window. "High share" is the percentage of that quarter's retained candles assigned to High.

| Quarter | Model | Mean rolling volatility, L / M / H (%) | High share (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.1273 / 0.2517 / 0.5029 | 15.14 | 8.02 | 100.00 |
| Q1 | B | 0.1335 / 0.2563 / 0.5028 | 14.51 | 11.14 | 86.51 |
| Q1 | C | 0.1211 / 0.2456 / 0.5348 | 16.63 | 4.66 | 82.63 |
| Q1 | D | 0.1246 / 0.2468 / 0.5313 | 15.77 | 7.78 | 84.52 |
| Q2 | A | 0.1137 / 0.2060 / 0.3802 | 12.11 | 8.91 | 100.00 |
| Q2 | B | 0.1175 / 0.1969 / 0.3516 | 14.66 | 14.05 | 83.29 |
| Q2 | C | 0.1005 / 0.1940 / 0.3967 | 15.22 | 4.94 | 79.54 |
| Q2 | D | 0.1015 / 0.1892 / 0.3727 | 17.05 | 8.71 | 80.06 |
| Q3 | A | 0.0886 / 0.1570 / 0.2457 | 12.76 | 12.35 | 100.00 |
| Q3 | B | 0.0965 / 0.1642 / 0.2409 | 10.94 | 15.02 | 79.90 |
| Q3 | C | 0.0757 / 0.1394 / 0.2455 | 23.54 | 5.96 | 72.59 |
| Q3 | D | 0.0849 / 0.1560 / 0.2664 | 13.76 | 9.62 | 78.81 |
| Q4 | A | 0.1165 / 0.2153 / 0.3702 | 18.92 | 11.54 | 100.00 |
| Q4 | B | 0.1281 / 0.2255 / 0.3720 | 15.74 | 15.89 | 80.23 |
| Q4 | C | 0.1056 / 0.2047 / 0.3871 | 22.74 | 5.93 | 78.33 |
| Q4 | D | 0.1138 / 0.2158 / 0.3973 | 18.49 | 9.43 | 80.25 |

Several concrete differences follow from these BTC 15m results:

- **State magnitudes vary by quarter.** A's High-state average rolling volatility is 0.5029% in Q1 and 0.2457% in Q3. Its High-state mean absolute log return also falls from 0.4667% to 0.2610%. The label High is relative to the quarter's fit, so it does not imply the same absolute movement level throughout the year.
- **Feature combinations change temporal behavior.** B switches more frequently than A in all four quarters, while C switches less frequently than A. D lies between B and C in every quarter. This describes the decoded sequences; deciding which behavior is preferable requires evaluating the associated regime separation and responsiveness.
- **The size of the change varies by quarter.** A-B agreement is 86.51% in Q1 and 79.90% in Q3. C's High-state share rises from 16.63% in Q1 to 23.54% in Q3, although C's High-state average volatility is lower in Q3. A higher High-state share alone therefore cannot establish that the entire quarter is more volatile.

These observations describe quarterly variation for the BTC 15m example. The full tables extend the same measurements to ETH, DOGE, LTC, and the other intervals. Improvement should be judged against explicitly chosen criteria; a lower switching frequency or greater agreement with A alone cannot select the best model. Historical rankings in `findings.txt` used earlier signed-return inputs and should not be substituted for the current results.

## How to define and evaluate performance

For these SOPs, define **descriptive regime quality** through the following complementary measures. The guide's complete tables provide annual state means, shares, durations and self-transition probabilities, plus quarterly state means, shares, agreement and switching for all 192 combinations. The quarterly summary contains feature distribution statistics. Distribution-overlap analysis, repeated-fit stability, and held-out predictive evaluation remain to be performed.

| Measure | Calculation or procedure | What it answers |
| --- | --- | --- |
| Feature profile within each state | For absolute return, range, volume, and rolling volatility, report count, mean, median, standard deviation, and interquartile range. Also report signed-return standard deviation. | What kinds of observations belong to each regime? |
| Separation and overlap | Compare the full distributions and both adjacent pairs: Low-Medium and Medium-High. A descriptive standardized mean gap is `abs(mean_i - mean_j) / sqrt((variance_i + variance_j) / 2)` when the denominator is positive. | Are the groups distinct beyond having ordered averages? |
| Occupancy | `100 * candles_in_state / valid_candles`. | Is each state represented by enough observations to interpret? The existing 1% screen is a modeling rule, not proof of a natural market boundary. |
| Persistence and switching | Report median consecutive run length in candles and minutes, switch rate, and learned self-transition probabilities. Break observed runs at missing intervals and quarter boundaries. | Do the states capture recurring periods, and how often do they change? |
| Agreement and disagreement | Align timestamps and ordered labels; report label agreement and a 3-by-3 cross-tabulation for each model pair. | Which observations are reclassified when a feature is added? Agreement is not accuracy. |
| Fit reliability and stability | Record errors, occupancy failures, stopping behavior, and sensitivity to repeated initializations under the same fitting rules. | Are conclusions robust to numerical fitting choices? The annual trainer's fallback seeds are not a completed stability experiment. |

Apply the same measures to all four models using the same valid candles in each comparison. Keep three states, full covariance, feature definitions, exclusions, and fitting settings consistent **within** the quarterly experiment. Report all four features even for a model that does not use a particular one. For volume, use its base-asset unit within each coin; raw BTC volume and raw DOGE volume are not directly comparable quantities.

The gap statistic is descriptive, not a significance test. Means can differ while distributions overlap considerably. Absolute return and range are inputs to every model, and rolling volatility defines the label order and is an input to C/D, so separation on these variables is internal evidence. It should not be described as independent validation. Likewise, extremely persistent states can miss short changes, and excessive switching can fragment regimes. No single one of these statistics provides a universal model ranking.

### Why log-likelihood alone cannot choose the best feature combination

The displayed log-likelihood measures how a model fits its **entire feature vector**. A fits two features, B and C each fit three different features, and D fits four. These scores refer to different observed variables. Dividing by the number of candles, averaging across quarters, or applying AIC/BIC penalties does not make them scores for the same prediction task. They should not be the sole basis for declaring an added feature effective. B and C having the same number of features does not solve the difference in what they model.

There are no known true Low/Medium/High labels in the raw candles. The current-state probability is the model's conditional state probability, not the percentage of classifications that are correct. A separately defined reference is required before using accuracy, precision, or recall as regime-detection claims.

### If the thesis also claims predictive improvement

Use a chronological training/evaluation split or a rolling evaluation design. Fit the scaler, HMM, and state ordering using training observations only. At each evaluation timestamp, form a prediction using only information then available. Score the same target for every model, such as the next candle's absolute logarithmic return, with the same loss and evaluation timestamps. Carry sufficient past candles into the first evaluation rolling window without using future data.

For a quarterly predictive study, apply the same chronological protocol within each quarter or explicitly train on preceding data and evaluate the following quarter. These are different designs; specify one before comparing results. A model trained on all of 2025 and summarized by 2025 quarter is an in-sample description and cannot serve as an unseen quarterly test.

Predicting return magnitude would evaluate a common volatility-related target. It would not create true regime labels. Historical Viterbi paths and smoothed posterior assignments use the supplied sequence; evaluation must not present those historical assignments as predictions made with only information available at each earlier candle. This predictive evaluation has not been completed for the guide.

## Sample definition and limits to disclose

The examples use the files in `src/HMM/data/raw/`. The stored quarter boundaries correspond to calendar quarters in **UTC+8**: the complete stored year runs from `2024-12-31 16:00 UTC` inclusive to `2025-12-31 16:00 UTC` exclusive. The timestamps are decoded in UTC by the analysis code. This should be stated explicitly in the methodology; the fetcher constructs naive local datetimes, so reproducing the fetch in another timezone can change the boundaries.

Derived features are calculated before excluding the interval **10 October 2025 at 20:00 UTC through 11 October 2025 at 04:00 UTC, inclusive**. The code introduced this exclusion to address collapsed states around an extreme event. The reported results consequently describe the retained sample and do not fully assess regime detection during that excluded event. Because rolling features are calculated first, some following windows still contain returns from the excluded interval.

The annual fits have 105,013 usable rows per coin at 5m, 34,997 at 15m, and 17,493 at 30m. Separate quarterly fitting drops the ten initial warmup rows of every quarter, whereas annual fitting drops them once after concatenation. Annual and quarterly fits therefore differ in both training scope and the retained observations. The current HMM training also treats retained rows across the excluded event as adjacent in its input sequence; the observed switch rates in this guide explicitly omit that gap.

The quarterly pipeline allows up to 500 training iterations with seed 42. The annual saved-model trainer allows up to 100 iterations and tries seeds 42, 7, and 2025 until it obtains an acceptable occupancy pattern. These are separate fitting procedures. Annual results, quarterly results, and live posterior-based assignments should be identified by their procedure rather than combined as interchangeable measurements.

The coverage update retrained all 48 annual saved models and reran the 192 quarterly combinations using the existing 2025 raw files. The guide updater then reads those fits and results to rebuild tables without training another set of models. The results and state statistics are stored with the processed outputs; no separate `reports/` directory is needed.

## Suggested Chapter 3 wording

The following paragraphs describe the implemented study design. The numerical observations and complete tables belong in the results discussion. Additional distribution-overlap, stability, and predictive evaluation should only be reported as completed after they have been carried out.

**For Objective 1:** This study characterizes the low, medium, and high volatility regimes identified by three-state Gaussian Hidden Markov Models in Bitcoin, Ethereum, Dogecoin, and Litecoin at 5-minute, 15-minute, and 30-minute trading intervals. Inputs are standardized using the relevant training sample, and inferred states are labeled by their mean rolling volatility. Each regime is described using statistics of absolute logarithmic returns, high-low price range, trading volume, and rolling volatility, together with signed-return dispersion, state frequency, duration, and transition behavior. These characteristics are compared across cryptocurrencies and trading intervals for each model. The use of three states is a predefined modeling choice.

**For Objective 2:** Four feature combinations are compared under a common three-state HMM structure with full covariance. Model A uses absolute logarithmic returns and normalized high-low price range; Model B adds trading volume; Model C adds rolling volatility; and Model D includes both additional variables. Comparisons examine changes in feature statistics within each state, state membership, persistence, and model agreement using identical observations. The comparisons of A with B and C with D assess the contribution associated with adding volume, while comparisons of A with C and B with D assess the contribution associated with adding rolling volatility. Joint log-likelihood across different feature sets is not treated as sufficient evidence of improved regime detection.

**For Objective 3:** Each feature combination is fitted independently to each quarterly sample to examine variation across time, cryptocurrencies, and trading intervals. The same descriptive evaluation criteria are applied to each of the 192 coin-interval-quarter-model combinations. State labels are aligned by their volatility ordering while retaining their numerical feature characteristics, since the same label can represent different movement levels in different quarters. Model failures and poorly populated states are reported alongside successful fits. Conclusions from these quarterly fits concern in-sample regime characteristics; claims about predictive performance require a separate chronological evaluation on unseen observations.

## Where to find each part in the code

| Study component | File and function or artifact |
| --- | --- |
| Raw candle source and quarter boundaries | [data_fetcher.py](src/HMM/data_fetcher.py): `QUARTERS`, `get_klines()`; stored inputs in `src/HMM/data/raw/`. |
| Return, range, volume, and rolling-volatility inputs | [feature_engineering.py](src/HMM/feature_engineering.py): `_base_features()`, `build_features_A()` through `build_features_D()`; dashboard equivalents in [server.py](src/HMM/server.py): `base_features()`, `MODEL_COLUMNS`, `MODELS`. |
| Gaussian model, decoding, label ordering, occupancy | [hmm_model.py](src/HMM/hmm_model.py): `train_hmm()`, `predict_states()`, `label_regimes()`, `is_degenerate()`. |
| Quarterly fitting and processed states | [main.py](src/HMM/main.py): `run()` exports all four coins and all intervals; `server.py`: `run_hmm()` supports the same selections. Per-candle outputs are in `src/HMM/data/processed/`. |
| State feature statistics, frequency and duration | [regime_statistics.py](src/HMM/regime_statistics.py): `summarize_regimes()`; quarterly statistics are saved in `hmm_results_summary.json`. |
| Complete guide tables | [update_model_guide.py](src/HMM/update_model_guide.py): reads all 48 annual models and all 192 quarterly outcomes. |
| Annual training and saved parameters | [train_2025.py](src/HMM/train_2025.py) calls `server.py`: `train_2025_models()`. Bundles are in `src/HMM/data/models/2025/`. |
| Current state probabilities and next-state distribution | `server.py`: `infer_saved_models()` uses causal forward filtering, saved scalers, closed-candle filter context, and the learned transition matrix. |
| Comparison charts, divergence summaries, and tooltips | [dashboard.html](src/HMM/dashboard.html): `modelAgreement()`, `windowForecastMetrics()`, and `updateComparisonAgreement()`. A remains the reference in A-B, A-C, and A-D. |
| Forecast probability alignment and divergence calculations | [forecast_comparison.py](src/HMM/forecast_comparison.py): `filter_probabilities()`, `align_regime_probabilities()`, `compare_distributions()`, and `summarize_comparisons()`. |

Standardization follows the [scikit-learn StandardScaler definition](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): subtract the training mean and divide by the training standard deviation. This rescales units; it does not assign a manually chosen importance to each feature or guarantee that feature distributions become Gaussian. The numerical findings in this guide come from the repository's data and models, rather than the library documentation.

<!-- BEGIN COMPLETE MODEL TABLES -->
## Complete annual regime profiles: every coin and interval

These tables cover all 48 annual saved models and all 144 decoded states. Each coin has 5m, 15m, and 30m tables for Models A-D. Values are training descriptions in original units. Volume is the amount of the named coin traded per candle. Return magnitude, range, and rolling volatility are percentages. All models use the same candles within a coin and interval.

Temporal summaries use observed consecutive runs, broken at missing candle intervals. Median run duration is in minutes. P(stay) is the learned self-transition probability for each ordered state, expressed as a percentage. Runs at the sample edges are truncated by the observation period.

<details>
<summary>BTC: annual profiles at 5m, 15m, and 30m</summary>

### BTC 5m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (BTC) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.0410 | 0.0680 | 28.07 | 0.0604 | 40.75 |
| A | Medium | 0.0897 | 0.1628 | 67.04 | 0.1154 | 44.92 |
| A | High | 0.2210 | 0.4094 | 218.85 | 0.2418 | 14.33 |
| B | Low | 0.0430 | 0.0728 | 23.71 | 0.0648 | 43.47 |
| B | Medium | 0.0925 | 0.1684 | 67.82 | 0.1184 | 42.89 |
| B | High | 0.2221 | 0.4077 | 245.70 | 0.2357 | 13.64 |
| C | Low | 0.0438 | 0.0745 | 30.43 | 0.0553 | 40.86 |
| C | Medium | 0.0880 | 0.1582 | 65.35 | 0.1110 | 41.67 |
| C | High | 0.1952 | 0.3614 | 190.31 | 0.2420 | 17.47 |
| D | Low | 0.0438 | 0.0743 | 25.34 | 0.0581 | 41.83 |
| D | Medium | 0.0897 | 0.1622 | 65.61 | 0.1144 | 42.60 |
| D | High | 0.2065 | 0.3811 | 220.70 | 0.2445 | 15.57 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 40.0 / 30.0 / 15.0 | 91.72 / 87.97 / 86.10 | 8.15 |
| B | 25.0 / 20.0 / 10.0 | 87.98 / 81.65 / 80.01 | 13.21 |
| C | 82.5 / 70.0 / 55.0 | 96.53 / 94.48 / 94.48 | 4.09 |
| D | 45.0 / 35.0 / 15.0 | 93.54 / 89.97 / 89.20 | 7.58 |

### BTC 15m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (BTC) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.0737 | 0.1386 | 89.41 | 0.1094 | 41.01 |
| A | Medium | 0.1567 | 0.3026 | 205.66 | 0.2066 | 44.90 |
| A | High | 0.3821 | 0.7360 | 636.73 | 0.4005 | 14.09 |
| B | Low | 0.0760 | 0.1445 | 76.77 | 0.1151 | 41.93 |
| B | Medium | 0.1567 | 0.3043 | 202.60 | 0.2070 | 43.76 |
| B | High | 0.3774 | 0.7173 | 683.84 | 0.3862 | 14.32 |
| C | Low | 0.0781 | 0.1484 | 96.09 | 0.0981 | 39.76 |
| C | Medium | 0.1517 | 0.2909 | 197.00 | 0.1938 | 42.19 |
| C | High | 0.3290 | 0.6355 | 539.79 | 0.4064 | 18.04 |
| D | Low | 0.0772 | 0.1461 | 81.49 | 0.1026 | 40.17 |
| D | Medium | 0.1531 | 0.2960 | 196.45 | 0.1980 | 43.32 |
| D | High | 0.3458 | 0.6631 | 611.26 | 0.4063 | 16.50 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 150.0 / 105.0 / 45.0 | 91.63 / 86.62 / 82.50 | 9.10 |
| B | 90.0 / 60.0 / 30.0 | 88.53 / 81.92 / 78.05 | 13.38 |
| C | 210.0 / 195.0 / 150.0 | 96.04 / 93.51 / 92.74 | 4.97 |
| D | 135.0 / 105.0 / 45.0 | 93.11 / 89.16 / 87.38 | 8.62 |

### BTC 30m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (BTC) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.0979 | 0.1995 | 179.75 | 0.1590 | 39.19 |
| A | Medium | 0.2127 | 0.4262 | 404.60 | 0.2921 | 45.70 |
| A | High | 0.5328 | 1.0314 | 1205.50 | 0.5272 | 15.11 |
| B | Low | 0.1014 | 0.2104 | 159.76 | 0.1675 | 40.76 |
| B | Medium | 0.2162 | 0.4327 | 402.67 | 0.2945 | 44.37 |
| B | High | 0.5303 | 1.0157 | 1302.43 | 0.5148 | 14.87 |
| C | Low | 0.1057 | 0.2138 | 191.23 | 0.1351 | 36.39 |
| C | Medium | 0.2017 | 0.4042 | 382.84 | 0.2625 | 42.74 |
| C | High | 0.4379 | 0.8540 | 978.75 | 0.5469 | 20.87 |
| D | Low | 0.1032 | 0.2119 | 168.31 | 0.1432 | 37.94 |
| D | Medium | 0.2101 | 0.4182 | 389.20 | 0.2741 | 44.11 |
| D | High | 0.4694 | 0.9135 | 1125.40 | 0.5586 | 17.94 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 210.0 / 180.0 / 90.0 | 88.83 / 83.05 / 78.01 | 12.20 |
| B | 180.0 / 120.0 / 60.0 | 86.53 / 79.48 / 74.95 | 15.65 |
| C | 390.0 / 330.0 / 300.0 | 94.83 / 91.81 / 91.37 | 6.36 |
| D | 240.0 / 180.0 / 90.0 | 91.79 / 87.64 / 86.34 | 9.90 |

</details>

<details>
<summary>ETH: annual profiles at 5m, 15m, and 30m</summary>

### ETH 5m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (ETH) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.0748 | 0.1436 | 801.64 | 0.1107 | 45.47 |
| A | Medium | 0.1555 | 0.3044 | 2032.18 | 0.2057 | 43.87 |
| A | High | 0.3956 | 0.7430 | 6850.03 | 0.3926 | 10.66 |
| B | Low | 0.0768 | 0.1491 | 675.33 | 0.1156 | 46.39 |
| B | Medium | 0.1555 | 0.3052 | 1984.41 | 0.2056 | 42.47 |
| B | High | 0.3835 | 0.7117 | 7455.27 | 0.3722 | 11.13 |
| C | Low | 0.0783 | 0.1514 | 845.89 | 0.0983 | 42.47 |
| C | Medium | 0.1462 | 0.2837 | 1864.09 | 0.1873 | 41.02 |
| C | High | 0.3100 | 0.5898 | 5222.26 | 0.3866 | 16.51 |
| D | Low | 0.0776 | 0.1498 | 703.20 | 0.1024 | 42.99 |
| D | Medium | 0.1484 | 0.2893 | 1850.93 | 0.1940 | 42.83 |
| D | High | 0.3346 | 0.6325 | 6282.92 | 0.3898 | 14.18 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 40.0 / 25.0 / 10.0 | 91.42 / 85.25 / 76.30 | 9.78 |
| B | 30.0 / 15.0 / 10.0 | 88.05 / 79.49 / 70.88 | 14.85 |
| C | 75.0 / 60.0 / 50.0 | 96.26 / 93.52 / 92.30 | 4.88 |
| D | 45.0 / 30.0 / 10.0 | 93.14 / 88.32 / 84.22 | 9.05 |

### ETH 15m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (ETH) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.1308 | 0.2732 | 2543.49 | 0.2021 | 45.72 |
| A | Medium | 0.2718 | 0.5601 | 6338.70 | 0.3640 | 44.07 |
| A | High | 0.6983 | 1.3438 | 19608.62 | 0.6400 | 10.21 |
| B | Low | 0.1349 | 0.2830 | 2226.23 | 0.2080 | 46.02 |
| B | Medium | 0.2664 | 0.5488 | 6016.22 | 0.3589 | 41.92 |
| B | High | 0.6394 | 1.2317 | 19988.58 | 0.5965 | 12.07 |
| C | Low | 0.1389 | 0.2882 | 2680.88 | 0.1748 | 41.80 |
| C | Medium | 0.2560 | 0.5229 | 5848.82 | 0.3336 | 42.90 |
| C | High | 0.5424 | 1.0727 | 15218.59 | 0.6663 | 15.30 |
| D | Low | 0.1367 | 0.2839 | 2306.15 | 0.1809 | 41.88 |
| D | Medium | 0.2530 | 0.5214 | 5650.47 | 0.3394 | 43.44 |
| D | High | 0.5706 | 1.1144 | 17295.02 | 0.6467 | 14.68 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 120.0 / 75.0 / 30.0 | 89.90 / 82.39 / 69.25 | 12.38 |
| B | 90.0 / 45.0 / 30.0 | 88.35 / 78.55 / 68.12 | 15.80 |
| C | 195.0 / 165.0 / 120.0 | 95.14 / 92.15 / 89.11 | 6.26 |
| D | 135.0 / 105.0 / 30.0 | 92.63 / 87.44 / 81.88 | 9.95 |

### ETH 30m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (ETH) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.1750 | 0.3926 | 5249.96 | 0.2963 | 44.91 |
| A | Medium | 0.3699 | 0.7879 | 12518.31 | 0.5176 | 43.71 |
| A | High | 0.9814 | 1.8689 | 35946.60 | 0.8354 | 11.38 |
| B | Low | 0.1843 | 0.4114 | 4699.56 | 0.3073 | 46.16 |
| B | Medium | 0.3703 | 0.7874 | 12300.96 | 0.5134 | 41.53 |
| B | High | 0.9181 | 1.7577 | 37682.80 | 0.8066 | 12.32 |
| C | Low | 0.1942 | 0.4277 | 5668.47 | 0.2506 | 42.14 |
| C | Medium | 0.3609 | 0.7577 | 12106.93 | 0.4781 | 41.89 |
| C | High | 0.7447 | 1.4763 | 27926.01 | 0.9300 | 15.97 |
| D | Low | 0.1912 | 0.4203 | 4948.26 | 0.2636 | 43.23 |
| D | Medium | 0.3621 | 0.7660 | 11956.53 | 0.4906 | 42.16 |
| D | High | 0.7981 | 1.5655 | 32436.13 | 0.9144 | 14.61 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 180.0 / 120.0 / 60.0 | 86.30 / 76.71 / 63.09 | 16.74 |
| B | 150.0 / 90.0 / 60.0 | 86.75 / 75.12 / 63.38 | 18.54 |
| C | 390.0 / 300.0 / 270.0 | 94.25 / 90.65 / 88.28 | 7.22 |
| D | 240.0 / 180.0 / 60.0 | 91.38 / 85.53 / 80.68 | 11.25 |

</details>

<details>
<summary>DOGE: annual profiles at 5m, 15m, and 30m</summary>

### DOGE 5m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (DOGE) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.1028 | 0.1995 | 1724541.20 | 0.1464 | 44.37 |
| A | Medium | 0.2024 | 0.3934 | 4352809.49 | 0.2613 | 44.07 |
| A | High | 0.4767 | 0.8976 | 15057109.69 | 0.4960 | 11.57 |
| B | Low | 0.1047 | 0.2071 | 1344327.39 | 0.1548 | 45.91 |
| B | Medium | 0.2057 | 0.3984 | 4312458.44 | 0.2631 | 42.58 |
| B | High | 0.4714 | 0.8772 | 17122959.21 | 0.4724 | 11.51 |
| C | Low | 0.1082 | 0.2127 | 1880579.85 | 0.1342 | 43.80 |
| C | Medium | 0.1972 | 0.3789 | 4147708.02 | 0.2476 | 40.35 |
| C | High | 0.3973 | 0.7550 | 12164444.69 | 0.4971 | 15.84 |
| D | Low | 0.1065 | 0.2092 | 1449862.61 | 0.1384 | 43.36 |
| D | Medium | 0.1979 | 0.3821 | 4066536.27 | 0.2525 | 42.45 |
| D | High | 0.4210 | 0.7947 | 14586834.26 | 0.4952 | 14.19 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 45.0 / 30.0 / 15.0 | 92.22 / 87.02 / 81.16 | 8.34 |
| B | 25.0 / 15.0 / 10.0 | 86.60 / 78.46 / 72.66 | 15.80 |
| C | 80.0 / 65.0 / 50.0 | 96.63 / 93.89 / 93.24 | 4.34 |
| D | 45.0 / 30.0 / 10.0 | 92.92 / 88.37 / 85.47 | 8.89 |

### DOGE 15m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (DOGE) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.1814 | 0.3672 | 5426628.51 | 0.2657 | 42.93 |
| A | Medium | 0.3476 | 0.7006 | 12942316.42 | 0.4584 | 44.76 |
| A | High | 0.8128 | 1.5649 | 41821967.72 | 0.7978 | 12.32 |
| B | Low | 0.1880 | 0.3839 | 4539524.60 | 0.2774 | 44.75 |
| B | Medium | 0.3522 | 0.7113 | 13056482.72 | 0.4634 | 43.40 |
| B | High | 0.8148 | 1.5477 | 47028938.92 | 0.7779 | 11.86 |
| C | Low | 0.1918 | 0.3916 | 5801449.06 | 0.2375 | 41.25 |
| C | Medium | 0.3359 | 0.6693 | 12205723.18 | 0.4245 | 40.99 |
| C | High | 0.6575 | 1.2841 | 33090707.37 | 0.8192 | 17.76 |
| D | Low | 0.1899 | 0.3872 | 4842705.18 | 0.2467 | 41.93 |
| D | Medium | 0.3383 | 0.6796 | 12244357.78 | 0.4360 | 42.37 |
| D | High | 0.7044 | 1.3606 | 38554661.00 | 0.8233 | 15.70 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 135.0 / 90.0 / 30.0 | 90.52 / 84.08 / 76.27 | 10.60 |
| B | 90.0 / 60.0 / 30.0 | 88.16 / 79.82 / 70.62 | 14.90 |
| C | 225.0 / 165.0 / 150.0 | 95.69 / 92.62 / 91.80 | 5.50 |
| D | 150.0 / 105.0 / 45.0 | 93.26 / 88.56 / 85.39 | 8.83 |

### DOGE 30m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (DOGE) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.1734 | 0.5520 | 12996535.75 | 0.4358 | 47.41 |
| A | Medium | 0.6089 | 0.9638 | 22773026.78 | 0.5874 | 31.85 |
| A | High | 0.9230 | 1.8522 | 63304137.32 | 0.9693 | 20.75 |
| B | Low | 0.2544 | 0.5514 | 9550252.86 | 0.4078 | 44.13 |
| B | Medium | 0.4890 | 1.0073 | 26225333.97 | 0.6600 | 43.39 |
| B | High | 1.1473 | 2.1834 | 87769711.69 | 1.0295 | 12.48 |
| C | Low | 0.2641 | 0.5660 | 12054080.19 | 0.3329 | 39.49 |
| C | Medium | 0.4596 | 0.9472 | 24478588.54 | 0.5951 | 40.93 |
| C | High | 0.8949 | 1.7448 | 60100215.37 | 1.1224 | 19.58 |
| D | Low | 0.2617 | 0.5605 | 10114397.55 | 0.3511 | 41.00 |
| D | Medium | 0.4684 | 0.9664 | 24867809.49 | 0.6215 | 42.67 |
| D | High | 0.9826 | 1.9024 | 72182614.01 | 1.1367 | 16.33 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 60.0 / 30.0 / 120.0 | 64.06 / 50.52 / 79.95 | 33.50 |
| B | 180.0 / 120.0 / 60.0 | 87.15 / 77.96 / 66.88 | 16.82 |
| C | 345.0 / 300.0 / 300.0 | 94.25 / 90.88 / 90.62 | 6.94 |
| D | 270.0 / 180.0 / 90.0 | 91.98 / 86.75 / 83.15 | 10.47 |

</details>

<details>
<summary>LTC: annual profiles at 5m, 15m, and 30m</summary>

### LTC 5m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (LTC) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.0582 | 0.1778 | 1054.92 | 0.1438 | 44.74 |
| A | Medium | 0.2226 | 0.3151 | 1702.38 | 0.1921 | 29.70 |
| A | High | 0.3170 | 0.6164 | 5674.24 | 0.3713 | 25.56 |
| B | Low | 0.0945 | 0.1830 | 731.49 | 0.1385 | 47.50 |
| B | Medium | 0.1919 | 0.3679 | 2348.48 | 0.2439 | 40.99 |
| B | High | 0.4316 | 0.8077 | 9711.85 | 0.4393 | 11.51 |
| C | Low | 0.0938 | 0.1821 | 949.96 | 0.1161 | 41.23 |
| C | Medium | 0.1743 | 0.3316 | 2143.55 | 0.2183 | 40.76 |
| C | High | 0.3525 | 0.6687 | 6454.87 | 0.4414 | 18.01 |
| D | Low | 0.0971 | 0.1866 | 791.24 | 0.1251 | 45.65 |
| D | Medium | 0.1856 | 0.3552 | 2262.28 | 0.2364 | 40.86 |
| D | High | 0.3933 | 0.7440 | 8468.95 | 0.4643 | 13.49 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 10.0 / 5.0 / 25.0 | 61.59 / 51.14 / 90.88 | 31.43 |
| B | 25.0 / 15.0 / 10.0 | 87.30 / 77.84 / 72.05 | 15.86 |
| C | 75.0 / 65.0 / 55.0 | 96.71 / 94.21 / 93.90 | 4.26 |
| D | 45.0 / 25.0 / 10.0 | 93.46 / 88.29 / 84.96 | 8.66 |

### LTC 15m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (LTC) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.1012 | 0.3372 | 3445.34 | 0.2615 | 43.61 |
| A | Medium | 0.3848 | 0.5570 | 5066.72 | 0.3370 | 30.97 |
| A | High | 0.5469 | 1.1006 | 16567.62 | 0.6319 | 25.43 |
| B | Low | 0.1688 | 0.3393 | 2456.29 | 0.2471 | 46.46 |
| B | Medium | 0.3277 | 0.6564 | 7165.50 | 0.4257 | 41.41 |
| B | High | 0.7271 | 1.4006 | 26181.88 | 0.7252 | 12.13 |
| C | Low | 0.0988 | 0.3455 | 3577.38 | 0.2536 | 41.14 |
| C | Medium | 0.3764 | 0.5391 | 4820.85 | 0.2915 | 29.15 |
| C | High | 0.5114 | 1.0098 | 14831.20 | 0.6385 | 29.71 |
| D | Low | 0.1693 | 0.3380 | 2539.51 | 0.2178 | 42.28 |
| D | Medium | 0.3120 | 0.6229 | 6661.94 | 0.4017 | 42.71 |
| D | High | 0.6492 | 1.2679 | 22409.42 | 0.7687 | 15.02 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 30.0 / 15.0 / 75.0 | 56.25 / 48.05 / 89.25 | 35.12 |
| B | 90.0 / 45.0 / 30.0 | 88.65 / 79.92 / 73.91 | 14.20 |
| C | 30.0 / 15.0 / 150.0 | 49.49 / 46.21 / 94.32 | 36.10 |
| D | 150.0 / 105.0 / 30.0 | 93.40 / 89.03 / 85.82 | 8.53 |

### LTC 30m: annual fitted states

| Model | State | Mean absolute return (%) | Mean range (%) | Mean volume (LTC) | Mean rolling volatility (%) | Share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| A | Low | 0.2195 | 0.4660 | 5870.09 | 0.3389 | 44.06 |
| A | Medium | 0.4539 | 0.9215 | 14626.21 | 0.6016 | 43.62 |
| A | High | 1.0557 | 2.0461 | 45471.10 | 1.0237 | 12.32 |
| B | Low | 0.2271 | 0.4830 | 5151.77 | 0.3512 | 45.25 |
| B | Medium | 0.4544 | 0.9293 | 14303.61 | 0.6015 | 42.18 |
| B | High | 1.0368 | 1.9788 | 49350.34 | 0.9962 | 12.57 |
| C | Low | 0.2336 | 0.4965 | 6282.12 | 0.2933 | 41.42 |
| C | Medium | 0.4325 | 0.8747 | 13571.15 | 0.5459 | 40.39 |
| C | High | 0.8425 | 1.6510 | 35639.55 | 1.0766 | 18.20 |
| D | Low | 0.2312 | 0.4908 | 5464.83 | 0.3056 | 42.54 |
| D | Medium | 0.4427 | 0.9001 | 13797.98 | 0.5653 | 42.01 |
| D | High | 0.9088 | 1.7628 | 41720.21 | 1.1027 | 15.45 |

| Model | Median run L / M / H (minutes) | P(stay) L / M / H (%) | Switch rate (%) |
| --- | --- | --- | --- |
| A | 240.0 / 150.0 / 90.0 | 90.26 / 82.71 / 74.52 | 11.53 |
| B | 180.0 / 120.0 / 60.0 | 88.20 / 79.55 / 72.95 | 14.84 |
| C | 420.0 / 300.0 / 300.0 | 95.42 / 91.92 / 91.78 | 5.99 |
| D | 300.0 / 240.0 / 90.0 | 93.22 / 88.72 / 87.12 | 8.49 |

</details>

## Complete quarterly comparison: every coin and interval

The batch attempted all 192 quarterly combinations: **192 successful fits and 0 failures**. Each row below is one independent quarterly fit. Triples are Low / Medium / High. Shares and switching describe the retained sample; agreement describes similarity to A. These measures do not establish predictive accuracy or a universal winner.

For every successful quarterly fit, [hmm_results_summary.json](src/HMM/data/processed/hmm_results_summary.json) also stores count, mean, median, standard deviation, first/third quartiles for all five feature columns, observed run lengths, and the transition matrix in regime order. Its `coin`, `timeframe`, `quarter`, and `model` fields identify each fit; `output_file` points to its candle-level JSON. Failed fits retain an explicit error and have no current candle output.

<details>
<summary>BTC: quarterly results at 5m, 15m, and 30m</summary>

### BTC 5m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.0530 / 0.1151 / 0.2817 | 0.0742 / 0.1466 / 0.3151 | 43.94 / 41.48 / 14.57 | 6.23 | 100.00 |
| Q1 | B | 0.0541 / 0.1169 / 0.2919 | 0.0775 / 0.1510 / 0.3108 | 44.83 / 41.83 / 13.34 | 11.06 | 85.11 |
| Q1 | C | 0.0540 / 0.1104 / 0.2485 | 0.0677 / 0.1376 / 0.3072 | 41.96 / 39.73 / 18.32 | 3.67 | 84.55 |
| Q1 | D | 0.0555 / 0.1148 / 0.2737 | 0.0716 / 0.1472 / 0.3204 | 44.62 / 40.44 / 14.94 | 7.07 | 84.39 |
| Q2 | A | 0.0267 / 0.1036 / 0.1714 | 0.0740 / 0.0896 / 0.1922 | 45.01 / 33.33 / 21.66 | 40.83 | 100.00 |
| Q2 | B | 0.0443 / 0.0885 / 0.2061 | 0.0665 / 0.1123 / 0.2148 | 45.18 / 41.93 / 12.89 | 15.00 | 57.21 |
| Q2 | C | 0.0442 / 0.0841 / 0.1805 | 0.0554 / 0.1060 / 0.2246 | 40.93 / 42.61 / 16.46 | 4.30 | 59.03 |
| Q2 | D | 0.0441 / 0.0843 / 0.1845 | 0.0580 / 0.1069 / 0.2204 | 41.58 / 42.38 / 16.04 | 8.12 | 59.15 |
| Q3 | A | 0.0210 / 0.0784 / 0.1164 | 0.0588 / 0.0684 / 0.1297 | 42.77 / 31.49 / 25.74 | 40.46 | 100.00 |
| Q3 | B | 0.0342 / 0.0712 / 0.1486 | 0.0525 / 0.0903 / 0.1474 | 44.00 / 44.75 / 11.25 | 15.33 | 51.75 |
| Q3 | C | 0.0337 / 0.0645 / 0.1242 | 0.0423 / 0.0821 / 0.1543 | 37.73 / 44.27 / 18.00 | 4.85 | 56.29 |
| Q3 | D | 0.0347 / 0.0675 / 0.1347 | 0.0455 / 0.0871 / 0.1575 | 40.37 / 45.69 / 13.94 | 8.11 | 53.41 |
| Q4 | A | 0.0424 / 0.0924 / 0.2157 | 0.0654 / 0.1232 / 0.2289 | 35.51 / 47.35 / 17.13 | 9.66 | 100.00 |
| Q4 | B | 0.0437 / 0.0991 / 0.2168 | 0.0718 / 0.1284 / 0.2222 | 39.63 / 44.52 / 15.85 | 15.42 | 80.42 |
| Q4 | C | 0.0470 / 0.0940 / 0.1912 | 0.0603 / 0.1197 / 0.2369 | 37.99 / 41.78 / 20.22 | 4.71 | 79.02 |
| Q4 | D | 0.0459 / 0.0946 / 0.1988 | 0.0627 / 0.1221 / 0.2345 | 37.56 / 43.73 / 18.71 | 8.35 | 80.73 |

### BTC 15m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.0881 / 0.1909 / 0.4667 | 0.1273 / 0.2517 / 0.5029 | 40.73 / 44.13 / 15.14 | 8.02 | 100.00 |
| Q1 | B | 0.0917 / 0.1959 / 0.4708 | 0.1335 / 0.2563 / 0.5028 | 43.13 / 42.36 / 14.51 | 11.14 | 86.51 |
| Q1 | C | 0.0968 / 0.1951 / 0.4288 | 0.1211 / 0.2456 / 0.5348 | 43.86 / 39.51 / 16.63 | 4.66 | 82.63 |
| Q1 | D | 0.0939 / 0.1956 / 0.4426 | 0.1246 / 0.2468 / 0.5313 | 43.04 / 41.19 / 15.77 | 7.78 | 84.52 |
| Q2 | A | 0.0781 / 0.1591 / 0.3695 | 0.1137 / 0.2060 / 0.3802 | 44.66 / 43.23 / 12.11 | 8.91 | 100.00 |
| Q2 | B | 0.0773 / 0.1521 / 0.3434 | 0.1175 / 0.1969 / 0.3516 | 42.42 / 42.92 / 14.66 | 14.05 | 83.29 |
| Q2 | C | 0.0810 / 0.1522 / 0.3222 | 0.1005 / 0.1940 / 0.3967 | 41.67 / 43.11 / 15.22 | 4.94 | 79.54 |
| Q2 | D | 0.0784 / 0.1460 / 0.3167 | 0.1015 / 0.1892 / 0.3727 | 39.47 / 43.48 / 17.05 | 8.71 | 80.06 |
| Q3 | A | 0.0584 / 0.1180 / 0.2610 | 0.0886 / 0.1570 / 0.2457 | 38.61 / 48.63 / 12.76 | 12.35 | 100.00 |
| Q3 | B | 0.0642 / 0.1265 / 0.2629 | 0.0965 / 0.1642 / 0.2409 | 45.31 / 43.75 / 10.94 | 15.02 | 79.90 |
| Q3 | C | 0.0609 / 0.1107 / 0.1958 | 0.0757 / 0.1394 / 0.2455 | 35.24 / 41.22 / 23.54 | 5.96 | 72.59 |
| Q3 | D | 0.0643 / 0.1202 / 0.2366 | 0.0849 / 0.1560 / 0.2664 | 41.16 / 45.08 / 13.76 | 9.62 | 78.81 |
| Q4 | A | 0.0739 / 0.1520 / 0.3690 | 0.1165 / 0.2153 / 0.3702 | 34.70 / 46.38 / 18.92 | 11.54 | 100.00 |
| Q4 | B | 0.0802 / 0.1640 / 0.3853 | 0.1281 / 0.2255 / 0.3720 | 39.24 / 45.02 / 15.74 | 15.89 | 80.23 |
| Q4 | C | 0.0830 / 0.1557 / 0.3164 | 0.1056 / 0.2047 / 0.3871 | 36.20 / 41.05 / 22.74 | 5.93 | 78.33 |
| Q4 | D | 0.0835 / 0.1604 / 0.3496 | 0.1138 / 0.2158 / 0.3973 | 38.26 / 43.25 / 18.49 | 9.43 | 80.25 |

### BTC 30m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.1232 / 0.2640 / 0.6680 | 0.1843 / 0.3762 / 0.6638 | 40.44 / 43.64 / 15.92 | 10.79 | 100.00 |
| Q1 | B | 0.1270 / 0.2653 / 0.6688 | 0.1927 / 0.3790 / 0.6550 | 42.32 / 41.67 / 16.01 | 12.79 | 87.84 |
| Q1 | C | 0.1361 / 0.2739 / 0.5568 | 0.1704 / 0.3439 / 0.7182 | 43.02 / 36.91 / 20.07 | 5.57 | 78.45 |
| Q1 | D | 0.1304 / 0.2670 / 0.5745 | 0.1728 / 0.3435 / 0.7044 | 40.84 / 39.61 / 19.56 | 8.38 | 82.58 |
| Q2 | A | 0.1084 / 0.2206 / 0.5535 | 0.1733 / 0.2982 / 0.5229 | 46.90 / 41.90 / 11.20 | 12.51 | 100.00 |
| Q2 | B | 0.1042 / 0.2008 / 0.4887 | 0.1722 / 0.2786 / 0.4716 | 40.48 / 44.38 / 15.14 | 17.14 | 82.74 |
| Q2 | C | 0.1065 / 0.1939 / 0.3991 | 0.1368 / 0.2472 / 0.5221 | 35.08 / 44.42 / 20.49 | 6.63 | 72.72 |
| Q2 | D | 0.1034 / 0.1870 / 0.3921 | 0.1391 / 0.2438 / 0.4923 | 34.19 / 42.96 / 22.85 | 9.25 | 72.03 |
| Q3 | A | 0.0757 / 0.1566 / 0.3756 | 0.1274 / 0.2197 / 0.3140 | 35.81 / 50.75 / 13.44 | 18.23 | 100.00 |
| Q3 | B | 0.0827 / 0.1642 / 0.3686 | 0.1364 / 0.2244 / 0.3086 | 40.63 / 46.66 / 12.71 | 19.66 | 82.68 |
| Q3 | C | 0.0811 / 0.1389 / 0.2604 | 0.1007 / 0.1841 / 0.3216 | 29.66 / 41.28 / 29.05 | 7.26 | 66.55 |
| Q3 | D | 0.0877 / 0.1671 / 0.3439 | 0.1197 / 0.2233 / 0.3653 | 39.85 / 47.93 / 12.21 | 11.44 | 76.71 |
| Q4 | A | 0.0874 / 0.2020 / 0.4945 | 0.1631 / 0.2973 / 0.4660 | 29.14 / 49.35 / 21.51 | 14.70 | 100.00 |
| Q4 | B | 0.1036 / 0.2233 / 0.5291 | 0.1823 / 0.3166 / 0.4744 | 36.27 / 46.84 / 16.88 | 18.19 | 78.24 |
| Q4 | C | 0.1168 / 0.2232 / 0.4353 | 0.1500 / 0.3022 / 0.5185 | 37.05 / 40.44 / 22.51 | 7.34 | 74.37 |
| Q4 | D | 0.1079 / 0.2209 / 0.4670 | 0.1498 / 0.3013 / 0.5253 | 33.88 / 46.25 / 19.87 | 10.30 | 77.26 |

</details>

<details>
<summary>ETH: quarterly results at 5m, 15m, and 30m</summary>

### ETH 5m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.0859 / 0.1781 / 0.4939 | 0.1205 / 0.2330 / 0.5053 | 49.51 / 42.05 / 8.44 | 7.65 | 100.00 |
| Q1 | B | 0.0883 / 0.1816 / 0.4888 | 0.1260 / 0.2359 / 0.4885 | 51.56 / 40.12 / 8.32 | 12.20 | 85.51 |
| Q1 | C | 0.0879 / 0.1709 / 0.3982 | 0.1089 / 0.2183 / 0.4976 | 46.62 / 41.57 / 11.81 | 4.25 | 83.05 |
| Q1 | D | 0.0872 / 0.1732 / 0.4312 | 0.1134 / 0.2238 / 0.5025 | 47.48 / 42.16 / 10.36 | 8.04 | 84.04 |
| Q2 | A | 0.0759 / 0.1541 / 0.3896 | 0.1144 / 0.2070 / 0.3875 | 42.83 / 44.90 / 12.27 | 11.61 | 100.00 |
| Q2 | B | 0.0793 / 0.1580 / 0.3987 | 0.1201 / 0.2123 / 0.3823 | 45.93 / 42.56 / 11.50 | 16.30 | 83.54 |
| Q2 | C | 0.0794 / 0.1450 / 0.3063 | 0.1004 / 0.1859 / 0.3839 | 39.67 / 41.40 / 18.93 | 5.36 | 77.84 |
| Q2 | D | 0.0797 / 0.1484 / 0.3339 | 0.1051 / 0.1948 / 0.3947 | 41.38 / 42.70 / 15.92 | 9.24 | 80.33 |
| Q3 | A | 0.0698 / 0.1348 / 0.3141 | 0.1029 / 0.1815 / 0.2936 | 41.67 / 46.61 / 11.72 | 11.20 | 100.00 |
| Q3 | B | 0.0711 / 0.1368 / 0.2958 | 0.1074 / 0.1799 / 0.2830 | 42.84 / 44.52 / 12.64 | 15.50 | 82.27 |
| Q3 | C | 0.0721 / 0.1283 / 0.2381 | 0.0902 / 0.1620 / 0.2987 | 38.40 / 41.52 / 20.08 | 5.35 | 74.90 |
| Q3 | D | 0.0718 / 0.1300 / 0.2509 | 0.0935 / 0.1673 / 0.2989 | 38.90 / 43.41 / 17.69 | 8.69 | 78.18 |
| Q4 | A | 0.0618 / 0.1387 / 0.3371 | 0.0981 / 0.1854 / 0.3532 | 38.16 / 46.13 / 15.71 | 11.13 | 100.00 |
| Q4 | B | 0.0643 / 0.1427 / 0.3335 | 0.1045 / 0.1907 / 0.3360 | 39.99 / 44.70 / 15.31 | 16.44 | 82.76 |
| Q4 | C | 0.0695 / 0.1352 / 0.2849 | 0.0881 / 0.1740 / 0.3573 | 38.40 / 41.16 / 20.44 | 4.72 | 78.83 |
| Q4 | D | 0.0656 / 0.1319 / 0.2922 | 0.0889 / 0.1743 / 0.3462 | 35.46 / 44.49 / 20.05 | 9.28 | 80.36 |

### ETH 15m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.1426 / 0.3011 / 0.8210 | 0.2125 / 0.4017 / 0.7919 | 47.10 / 43.81 / 9.08 | 10.86 | 100.00 |
| Q1 | B | 0.1504 / 0.2999 / 0.7977 | 0.2209 / 0.4053 / 0.7587 | 49.37 / 41.07 / 9.56 | 13.83 | 85.71 |
| Q1 | C | 0.1490 / 0.2773 / 0.6207 | 0.1870 / 0.3537 / 0.7934 | 42.72 / 42.36 / 14.91 | 5.90 | 79.57 |
| Q1 | D | 0.1502 / 0.2877 / 0.6798 | 0.1970 / 0.3714 / 0.8232 | 45.28 / 42.41 / 12.31 | 9.04 | 82.35 |
| Q2 | A | 0.1380 / 0.2680 / 0.6767 | 0.2114 / 0.3689 / 0.6254 | 44.03 / 43.04 / 12.93 | 14.12 | 100.00 |
| Q2 | B | 0.1416 / 0.2685 / 0.6505 | 0.2178 / 0.3647 / 0.6117 | 44.51 / 41.98 / 13.51 | 16.71 | 85.29 |
| Q2 | C | 0.1520 / 0.2708 / 0.5660 | 0.1882 / 0.3543 / 0.6917 | 43.98 / 40.75 / 15.26 | 6.46 | 75.60 |
| Q2 | D | 0.1484 / 0.2626 / 0.6063 | 0.1909 / 0.3590 / 0.6756 | 43.24 / 42.10 / 14.66 | 9.66 | 79.34 |
| Q3 | A | 0.1196 / 0.2332 / 0.5663 | 0.1847 / 0.3192 / 0.4794 | 40.73 / 48.13 / 11.14 | 14.03 | 100.00 |
| Q3 | B | 0.1240 / 0.2335 / 0.5038 | 0.1916 / 0.3155 / 0.4576 | 42.60 / 43.65 / 13.75 | 17.05 | 82.93 |
| Q3 | C | 0.1314 / 0.2282 / 0.4152 | 0.1620 / 0.2948 / 0.5145 | 40.18 / 41.24 / 18.58 | 6.68 | 73.89 |
| Q3 | D | 0.1257 / 0.2206 / 0.4291 | 0.1627 / 0.2930 / 0.4972 | 37.77 / 43.40 / 18.83 | 9.78 | 76.40 |
| Q4 | A | 0.1079 / 0.2372 / 0.5713 | 0.1813 / 0.3289 / 0.5585 | 38.33 / 45.14 / 16.53 | 12.91 | 100.00 |
| Q4 | B | 0.1094 / 0.2297 / 0.5517 | 0.1836 / 0.3253 / 0.5311 | 36.19 / 46.19 / 17.61 | 17.54 | 83.43 |
| Q4 | C | 0.1152 / 0.2183 / 0.4654 | 0.1494 / 0.2895 / 0.5715 | 32.42 / 44.11 / 23.47 | 6.54 | 77.36 |
| Q4 | D | 0.1169 / 0.2246 / 0.4937 | 0.1616 / 0.3052 / 0.5716 | 35.20 / 43.92 / 20.88 | 10.78 | 79.03 |

### ETH 30m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.2163 / 0.4613 / 1.2479 | 0.3239 / 0.6213 / 1.1289 | 52.13 / 40.67 / 7.19 | 11.77 | 100.00 |
| Q1 | B | 0.2206 / 0.4430 / 1.1716 | 0.3309 / 0.6041 / 1.0650 | 51.21 / 40.42 / 8.38 | 13.72 | 87.05 |
| Q1 | C | 0.2211 / 0.4185 / 0.8475 | 0.2751 / 0.5358 / 1.1338 | 45.03 / 40.86 / 14.11 | 5.34 | 76.91 |
| Q1 | D | 0.2202 / 0.4270 / 0.9294 | 0.2909 / 0.5576 / 1.1472 | 48.10 / 39.44 / 12.46 | 8.19 | 82.23 |
| Q2 | A | 0.1787 / 0.3646 / 1.0236 | 0.3080 / 0.5352 / 0.8658 | 43.23 / 43.78 / 12.99 | 20.84 | 100.00 |
| Q2 | B | 0.1883 / 0.3884 / 1.0009 | 0.3247 / 0.5396 / 0.8766 | 47.22 / 40.39 / 12.39 | 21.07 | 83.57 |
| Q2 | C | 0.2085 / 0.4010 / 0.8200 | 0.2669 / 0.5267 / 1.0567 | 44.84 / 42.01 / 13.15 | 7.09 | 73.15 |
| Q2 | D | 0.2015 / 0.3790 / 0.8662 | 0.2729 / 0.5192 / 1.0043 | 43.30 / 42.82 / 13.88 | 11.84 | 77.03 |
| Q3 | A | 0.1543 / 0.3001 / 0.7224 | 0.2695 / 0.4275 / 0.6175 | 36.13 / 48.77 / 15.09 | 19.91 | 100.00 |
| Q3 | B | 0.1665 / 0.3008 / 0.6602 | 0.2776 / 0.4258 / 0.6073 | 39.76 / 42.49 / 17.75 | 20.09 | 83.27 |
| Q3 | C | 0.1660 / 0.2836 / 0.5272 | 0.2097 / 0.3744 / 0.6618 | 30.16 / 43.96 / 25.87 | 7.49 | 66.93 |
| Q3 | D | 0.1698 / 0.2873 / 0.5930 | 0.2318 / 0.4078 / 0.6585 | 35.47 / 43.10 / 21.43 | 12.44 | 75.17 |
| Q4 | A | 0.1348 / 0.3058 / 0.8120 | 0.2578 / 0.4627 / 0.7093 | 34.02 / 48.19 / 17.79 | 17.21 | 100.00 |
| Q4 | B | 0.1455 / 0.3106 / 0.7441 | 0.2689 / 0.4604 / 0.6831 | 35.13 / 45.23 / 19.64 | 20.70 | 82.78 |
| Q4 | C | 0.1496 / 0.3022 / 0.6177 | 0.2050 / 0.4000 / 0.7698 | 29.57 / 44.86 / 25.56 | 8.18 | 75.55 |
| Q4 | D | 0.1440 / 0.2979 / 0.6577 | 0.2157 / 0.4203 / 0.7468 | 30.08 / 46.00 / 23.92 | 13.29 | 77.97 |

</details>

<details>
<summary>DOGE: quarterly results at 5m, 15m, and 30m</summary>

### DOGE 5m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.1251 / 0.2438 / 0.6058 | 0.1711 / 0.3110 / 0.6340 | 48.38 / 41.80 / 9.81 | 7.10 | 100.00 |
| Q1 | B | 0.1253 / 0.2412 / 0.5749 | 0.1779 / 0.3049 / 0.5909 | 48.22 / 40.81 / 10.97 | 14.19 | 83.16 |
| Q1 | C | 0.1286 / 0.2314 / 0.4882 | 0.1572 / 0.2887 / 0.6068 | 46.14 / 39.10 / 14.77 | 4.04 | 81.57 |
| Q1 | D | 0.1279 / 0.2343 / 0.5276 | 0.1628 / 0.2982 / 0.6132 | 46.74 / 40.52 / 12.74 | 8.42 | 83.08 |
| Q2 | A | 0.0642 / 0.2416 / 0.3453 | 0.1700 / 0.2028 / 0.3939 | 44.46 / 32.01 / 23.53 | 39.38 | 100.00 |
| Q2 | B | 0.1055 / 0.2016 / 0.4375 | 0.1580 / 0.2538 / 0.4394 | 45.77 / 41.71 / 12.52 | 18.36 | 58.42 |
| Q2 | C | 0.0637 / 0.2393 / 0.3186 | 0.1633 / 0.1829 / 0.3966 | 42.07 / 30.52 / 27.41 | 38.35 | 89.32 |
| Q2 | D | 0.1080 / 0.1901 / 0.3852 | 0.1391 / 0.2420 / 0.4550 | 42.21 / 41.52 / 16.27 | 9.37 | 59.62 |
| Q3 | A | 0.1026 / 0.1983 / 0.4434 | 0.1477 / 0.2547 / 0.4606 | 40.44 / 47.38 / 12.17 | 8.62 | 100.00 |
| Q3 | B | 0.1037 / 0.2040 / 0.4336 | 0.1565 / 0.2591 / 0.4322 | 43.01 / 44.57 / 12.42 | 16.05 | 81.15 |
| Q3 | C | 0.1119 / 0.1987 / 0.3778 | 0.1385 / 0.2495 / 0.4713 | 43.53 / 40.56 / 15.91 | 4.47 | 78.94 |
| Q3 | D | 0.1090 / 0.1984 / 0.4052 | 0.1417 / 0.2544 / 0.4679 | 42.06 / 44.12 / 13.82 | 8.93 | 80.33 |
| Q4 | A | 0.0790 / 0.1664 / 0.3944 | 0.1208 / 0.2195 / 0.4135 | 40.47 / 46.02 / 13.52 | 10.00 | 100.00 |
| Q4 | B | 0.0825 / 0.1724 / 0.4013 | 0.1296 / 0.2260 / 0.4010 | 44.01 / 43.31 / 12.68 | 17.27 | 82.70 |
| Q4 | C | 0.0860 / 0.1606 / 0.3322 | 0.1090 / 0.2055 / 0.4196 | 39.60 / 42.43 / 17.96 | 4.76 | 80.49 |
| Q4 | D | 0.0863 / 0.1672 / 0.3572 | 0.1155 / 0.2164 / 0.4272 | 42.17 / 42.68 / 15.15 | 9.28 | 81.53 |

### DOGE 15m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.2179 / 0.4146 / 1.0612 | 0.3083 / 0.5389 / 1.0353 | 47.31 / 42.79 / 9.90 | 9.20 | 100.00 |
| Q1 | B | 0.2231 / 0.4187 / 1.0631 | 0.3229 / 0.5446 / 0.9956 | 50.09 / 39.85 / 10.06 | 13.90 | 82.89 |
| Q1 | C | 0.2148 / 0.3758 / 0.7994 | 0.2653 / 0.4690 / 0.9908 | 39.40 / 43.33 / 17.28 | 5.26 | 78.08 |
| Q1 | D | 0.2226 / 0.4025 / 0.9043 | 0.2864 / 0.5115 / 1.0527 | 46.04 / 40.83 / 13.13 | 8.23 | 80.50 |
| Q2 | A | 0.1890 / 0.3554 / 0.7573 | 0.2740 / 0.4550 / 0.7687 | 44.44 / 42.76 / 12.80 | 10.04 | 100.00 |
| Q2 | B | 0.1976 / 0.3742 / 0.7810 | 0.2929 / 0.4683 / 0.7596 | 49.05 / 39.81 / 11.14 | 14.99 | 81.39 |
| Q2 | C | 0.1976 / 0.3359 / 0.6247 | 0.2408 / 0.4199 / 0.7870 | 40.34 / 41.39 / 18.27 | 5.48 | 75.84 |
| Q2 | D | 0.1990 / 0.3484 / 0.6655 | 0.2583 / 0.4348 / 0.7984 | 44.60 / 39.27 / 16.12 | 8.64 | 80.12 |
| Q3 | A | 0.1144 / 0.4055 / 0.5837 | 0.3177 / 0.3644 / 0.6415 | 40.18 / 33.18 / 26.64 | 40.58 | 100.00 |
| Q3 | B | 0.1922 / 0.3448 / 0.7445 | 0.2848 / 0.4545 / 0.7114 | 41.06 / 45.47 / 13.48 | 15.86 | 54.94 |
| Q3 | C | 0.1892 / 0.3098 / 0.5618 | 0.2303 / 0.3854 / 0.7108 | 31.72 / 42.70 / 25.58 | 5.82 | 58.57 |
| Q3 | D | 0.1959 / 0.3287 / 0.6340 | 0.2499 / 0.4231 / 0.7500 | 37.01 / 44.50 / 18.49 | 9.09 | 56.50 |
| Q4 | A | 0.1433 / 0.2775 / 0.6723 | 0.2203 / 0.3846 / 0.6489 | 39.04 / 46.83 / 14.13 | 11.88 | 100.00 |
| Q4 | B | 0.1456 / 0.2745 / 0.6443 | 0.2241 / 0.3824 / 0.6232 | 38.64 / 46.16 / 15.20 | 16.37 | 84.33 |
| Q4 | C | 0.1608 / 0.2779 / 0.5552 | 0.2015 / 0.3655 / 0.6903 | 41.78 / 39.49 / 18.73 | 5.76 | 75.57 |
| Q4 | D | 0.1555 / 0.2764 / 0.5911 | 0.2072 / 0.3689 / 0.6893 | 40.63 / 42.33 / 17.04 | 10.38 | 79.80 |

### DOGE 30m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.2891 / 0.5912 / 1.4530 | 0.4479 / 0.7787 / 1.3332 | 44.15 / 44.76 / 11.09 | 12.72 | 100.00 |
| Q1 | B | 0.3069 / 0.5957 / 1.4926 | 0.4719 / 0.7893 / 1.3156 | 47.52 / 41.90 / 10.58 | 15.90 | 82.46 |
| Q1 | C | 0.3080 / 0.5528 / 1.0723 | 0.3817 / 0.6904 / 1.3613 | 39.88 / 41.21 / 18.91 | 5.89 | 72.44 |
| Q1 | D | 0.3061 / 0.5461 / 1.1870 | 0.4039 / 0.7108 / 1.3725 | 40.95 / 42.58 / 16.47 | 9.82 | 79.28 |
| Q2 | A | 0.2621 / 0.4789 / 1.0869 | 0.4171 / 0.6413 / 1.0267 | 45.27 / 40.89 / 13.84 | 14.60 | 100.00 |
| Q2 | B | 0.2716 / 0.4916 / 1.1158 | 0.4289 / 0.6634 / 1.0037 | 48.60 / 38.55 / 12.85 | 19.65 | 82.54 |
| Q2 | C | 0.2713 / 0.4482 / 0.8255 | 0.3378 / 0.5689 / 1.0727 | 37.68 / 40.22 / 22.10 | 6.31 | 69.96 |
| Q2 | D | 0.2780 / 0.4631 / 0.8883 | 0.3679 / 0.5947 / 1.0966 | 44.31 / 35.98 / 19.71 | 10.40 | 75.10 |
| Q3 | A | 0.1754 / 0.6018 / 0.8702 | 0.4759 / 0.5352 / 0.8813 | 47.68 / 28.85 / 23.47 | 42.84 | 100.00 |
| Q3 | B | 0.2395 / 0.4286 / 0.9874 | 0.3936 / 0.6153 / 0.8907 | 33.73 / 48.98 / 17.29 | 18.52 | 57.13 |
| Q3 | C | 0.2783 / 0.4504 / 0.7952 | 0.3434 / 0.5840 / 1.0160 | 39.04 / 38.27 / 22.70 | 7.38 | 54.31 |
| Q3 | D | 0.2602 / 0.4307 / 0.8753 | 0.3486 / 0.5920 / 0.9926 | 35.16 / 44.44 / 20.40 | 11.65 | 55.54 |
| Q4 | A | 0.1834 / 0.3792 / 0.9583 | 0.3125 / 0.5437 / 0.8678 | 37.50 / 47.44 / 15.06 | 13.72 | 100.00 |
| Q4 | B | 0.1966 / 0.3755 / 0.9293 | 0.3216 / 0.5475 / 0.8416 | 38.98 / 45.27 / 15.74 | 15.98 | 83.37 |
| Q4 | C | 0.1969 / 0.3450 / 0.7219 | 0.2551 / 0.4680 / 0.8864 | 31.90 / 42.83 / 25.27 | 7.61 | 74.53 |
| Q4 | D | 0.2028 / 0.3711 / 0.7867 | 0.2791 / 0.5084 / 0.9145 | 37.02 / 42.72 / 20.26 | 10.78 | 78.15 |

</details>

<details>
<summary>LTC: quarterly results at 5m, 15m, and 30m</summary>

### LTC 5m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.1299 / 0.2424 / 0.5904 | 0.1722 / 0.3206 / 0.6044 | 42.94 / 46.09 / 10.98 | 6.21 | 100.00 |
| Q1 | B | 0.1286 / 0.2434 / 0.5647 | 0.1805 / 0.3188 / 0.5620 | 43.43 / 44.51 / 12.05 | 14.62 | 83.97 |
| Q1 | C | 0.1308 / 0.2399 / 0.4775 | 0.1600 / 0.2974 / 0.5978 | 42.23 / 41.57 / 16.19 | 3.87 | 81.94 |
| Q1 | D | 0.1307 / 0.2402 / 0.5088 | 0.1656 / 0.3052 / 0.5982 | 42.32 / 43.37 / 14.31 | 8.09 | 83.47 |
| Q2 | A | 0.0901 / 0.1656 / 0.3692 | 0.1268 / 0.2109 / 0.3776 | 43.99 / 45.26 / 10.75 | 8.60 | 100.00 |
| Q2 | B | 0.0923 / 0.1763 / 0.3704 | 0.1380 / 0.2161 / 0.3581 | 49.07 / 41.05 / 9.87 | 19.45 | 79.10 |
| Q2 | C | 0.0516 / 0.1964 / 0.2610 | 0.1380 / 0.1526 / 0.3241 | 40.85 / 32.80 / 26.35 | 39.30 | 52.85 |
| Q2 | D | 0.0963 / 0.1667 / 0.3347 | 0.1232 / 0.2115 / 0.3888 | 46.72 / 41.09 / 12.19 | 9.44 | 79.45 |
| Q3 | A | 0.0489 / 0.1863 / 0.2541 | 0.1275 / 0.1585 / 0.2943 | 43.27 / 29.51 / 27.22 | 34.69 | 100.00 |
| Q3 | B | 0.0833 / 0.1636 / 0.3558 | 0.1225 / 0.2106 / 0.3420 | 48.12 / 41.30 / 10.57 | 16.82 | 56.01 |
| Q3 | C | 0.0822 / 0.1459 / 0.2786 | 0.1014 / 0.1848 / 0.3486 | 40.88 / 39.96 / 19.16 | 4.54 | 60.83 |
| Q3 | D | 0.0831 / 0.1529 / 0.3024 | 0.1073 / 0.1961 / 0.3576 | 43.30 / 41.58 / 15.12 | 9.29 | 58.59 |
| Q4 | A | 0.0840 / 0.1854 / 0.4091 | 0.1272 / 0.2349 / 0.4245 | 48.26 / 40.27 / 11.46 | 10.51 | 100.00 |
| Q4 | B | 0.0850 / 0.1874 / 0.4152 | 0.1320 / 0.2364 / 0.4135 | 48.65 / 40.58 / 10.77 | 18.05 | 83.80 |
| Q4 | C | 0.0880 / 0.1663 / 0.3316 | 0.1104 / 0.2116 / 0.4168 | 42.54 / 39.85 / 17.61 | 4.51 | 79.00 |
| Q4 | D | 0.0894 / 0.1761 / 0.3709 | 0.1174 / 0.2274 / 0.4371 | 45.80 / 41.02 / 13.18 | 9.14 | 81.73 |

### LTC 15m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.2208 / 0.4077 / 0.9868 | 0.3024 / 0.5541 / 0.9664 | 41.01 / 47.46 / 11.53 | 8.62 | 100.00 |
| Q1 | B | 0.2208 / 0.4096 / 0.9269 | 0.3162 / 0.5496 / 0.9054 | 41.56 / 45.55 / 12.89 | 13.49 | 84.70 |
| Q1 | C | 0.2304 / 0.4093 / 0.7976 | 0.2786 / 0.5130 / 1.0249 | 41.11 / 42.91 / 15.98 | 4.77 | 79.86 |
| Q1 | D | 0.2244 / 0.4010 / 0.8033 | 0.2815 / 0.5115 / 0.9813 | 39.32 / 44.22 / 16.47 | 7.82 | 81.41 |
| Q2 | A | 0.1600 / 0.2939 / 0.6515 | 0.2325 / 0.3744 / 0.6509 | 43.69 / 45.54 / 10.77 | 9.46 | 100.00 |
| Q2 | B | 0.1627 / 0.2957 / 0.6205 | 0.2410 / 0.3787 / 0.5935 | 44.44 / 44.06 / 11.49 | 16.58 | 81.86 |
| Q2 | C | 0.1662 / 0.2821 / 0.5342 | 0.2018 / 0.3580 / 0.6674 | 41.03 / 43.35 / 15.62 | 5.55 | 75.12 |
| Q2 | D | 0.1656 / 0.2829 / 0.5574 | 0.2091 / 0.3658 / 0.6521 | 41.37 / 44.22 / 14.41 | 9.13 | 78.75 |
| Q3 | A | 0.1436 / 0.2616 / 0.5564 | 0.2068 / 0.3532 / 0.5462 | 41.10 / 44.25 / 14.65 | 11.52 | 100.00 |
| Q3 | B | 0.1467 / 0.2720 / 0.5497 | 0.2176 / 0.3557 / 0.5369 | 42.84 / 43.50 / 13.66 | 17.38 | 81.94 |
| Q3 | C | 0.0752 / 0.2893 / 0.3873 | 0.2179 / 0.2281 / 0.4854 | 32.61 / 29.88 / 37.51 | 35.91 | 48.49 |
| Q3 | D | 0.1505 / 0.2665 / 0.4923 | 0.1941 / 0.3449 / 0.5789 | 42.33 / 40.48 / 17.20 | 9.55 | 78.98 |
| Q4 | A | 0.1464 / 0.3040 / 0.6940 | 0.2206 / 0.4032 / 0.6786 | 44.81 / 42.39 / 12.80 | 10.21 | 100.00 |
| Q4 | B | 0.1520 / 0.3094 / 0.7023 | 0.2290 / 0.4068 / 0.6799 | 46.73 / 41.20 / 12.07 | 15.42 | 84.07 |
| Q4 | C | 0.1462 / 0.2724 / 0.5432 | 0.1841 / 0.3463 / 0.6770 | 36.20 / 42.91 / 20.89 | 5.18 | 76.68 |
| Q4 | D | 0.1564 / 0.3018 / 0.6220 | 0.2086 / 0.3872 / 0.7281 | 45.55 / 39.55 / 14.90 | 9.45 | 80.67 |

### LTC 30m: independent quarterly fits

| Quarter | Model | Mean absolute return L / M / H (%) | Mean rolling volatility L / M / H (%) | Shares L / M / H (%) | Switch rate (%) | Agreement with A (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | A | 0.3193 / 0.5942 / 1.5079 | 0.4411 / 0.8371 / 1.2978 | 42.32 / 47.68 / 10.00 | 10.21 | 100.00 |
| Q1 | B | 0.3150 / 0.5599 / 1.2529 | 0.4595 / 0.7786 / 1.1954 | 39.93 / 44.62 / 15.45 | 15.53 | 83.32 |
| Q1 | C | 0.3227 / 0.5786 / 1.0530 | 0.3933 / 0.7205 / 1.3633 | 39.93 / 40.51 / 19.56 | 5.20 | 76.94 |
| Q1 | D | 0.3064 / 0.5409 / 1.0478 | 0.3871 / 0.6947 / 1.2854 | 35.22 / 42.90 / 21.88 | 8.42 | 75.64 |
| Q2 | A | 0.2105 / 0.4147 / 0.9461 | 0.3362 / 0.5361 / 0.8668 | 43.60 / 46.03 / 10.37 | 14.05 | 100.00 |
| Q2 | B | 0.2159 / 0.4158 / 0.8740 | 0.3443 / 0.5401 / 0.7965 | 44.40 / 43.87 / 11.73 | 19.00 | 83.85 |
| Q2 | C | 0.2297 / 0.3928 / 0.6976 | 0.2863 / 0.4887 / 0.9195 | 40.96 / 41.07 / 17.97 | 6.33 | 71.68 |
| Q2 | D | 0.2293 / 0.4175 / 0.7937 | 0.3066 / 0.5279 / 0.9586 | 45.07 / 42.15 / 12.78 | 10.97 | 76.71 |
| Q3 | A | 0.0961 / 0.3778 / 0.5838 | 0.3266 / 0.3756 / 0.6389 | 32.77 / 33.43 / 33.79 | 37.98 | 100.00 |
| Q3 | B | 0.2068 / 0.3926 / 0.7963 | 0.3202 / 0.5130 / 0.7297 | 47.68 / 39.65 / 12.66 | 17.73 | 49.39 |
| Q3 | C | 0.2083 / 0.3661 / 0.6377 | 0.2576 / 0.4590 / 0.8236 | 40.90 / 39.38 / 19.72 | 5.93 | 52.04 |
| Q3 | D | 0.2069 / 0.3545 / 0.6314 | 0.2645 / 0.4561 / 0.7773 | 40.63 / 37.49 / 21.88 | 9.08 | 53.81 |
| Q4 | A | 0.1966 / 0.4206 / 1.0309 | 0.3211 / 0.5696 / 0.9581 | 44.32 / 43.27 / 12.42 | 13.54 | 100.00 |
| Q4 | B | 0.2023 / 0.4253 / 0.9747 | 0.3297 / 0.5670 / 0.9303 | 45.25 / 41.51 / 13.24 | 16.69 | 84.39 |
| Q4 | C | 0.2248 / 0.4297 / 0.8910 | 0.2879 / 0.5590 / 1.1091 | 46.48 / 39.94 / 13.58 | 6.66 | 77.03 |
| Q4 | D | 0.2174 / 0.4228 / 0.8886 | 0.2904 / 0.5542 / 1.0591 | 45.27 / 40.28 / 14.45 | 9.76 | 77.76 |

</details>

<!-- END COMPLETE MODEL TABLES -->
