"""
volume_analysis.py
Thesis evidence that volume tracks volatility regimes in EVERY quarter.

The LL/AIC/BIC comparison in main.py / the dashboard rewards features with
heavy tails and high autocorrelation, so raw volume only "wins" in quarters
with extreme volume spikes (Q1/Q4 2025). This script makes the volume case
the valid way instead:

  1. Fit Model C (log return + high-low range + rolling vol) — NO volume.
  2. Decode the hidden regimes.
  3. Measure mean deseasonalized volume surprise (volume_z) inside each regime.

If volume carries regime information, high-vol regimes should show elevated
volume surprise and low-vol regimes depressed volume surprise — out of
sample with respect to the features the HMM was trained on.

Usage:
    python volume_analysis.py            # BTC, 15m
    python volume_analysis.py 5m         # BTC, other timeframe
"""

import sys
import numpy as np
import pandas as pd

from main import find_file
from data_loader import load_binance_json
from feature_engineering import MODEL_BUILDERS
from hmm_model import train_hmm, predict_states, label_regimes

QUARTERS        = ["Q1", "Q2", "Q3", "Q4"]
VOL_WINDOW      = 10
SEASONAL_DAYS   = 20   # sessions used for the time-of-day volume baseline
VOLUME_Z_WINDOW = 50   # bars used to z-score deseasonalized volume


def add_volume_surprise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deseasonalized volume surprise: log volume minus its time-of-day
    rolling-median baseline, z-scored over a rolling window. Measures
    "how abnormal is volume right now" independent of clock time and drift.
    """
    df = df.copy()
    log_vol = np.log1p(df["volume"])
    tod = df["time"].dt.time
    seasonal = log_vol.groupby(tod).transform(
        lambda s: s.shift(1).rolling(SEASONAL_DAYS, min_periods=5).median()
    )
    deseason = log_vol - seasonal
    roll = deseason.rolling(VOLUME_Z_WINDOW)
    df["volume_z"] = (deseason - roll.mean()) / roll.std()
    return df


def run(tf: str = "15m"):
    print(f"Volume surprise by regime — regimes fitted WITHOUT volume (Model C, BTC {tf})")
    print(f"{'Q':<4}{'Low (Bull)':>12}{'Medium':>12}{'High (Bear)':>13}{'monotone':>10}")

    for q in QUARTERS:
        path = find_file(tf, q)
        if path is None:
            print(f"{q:<4}  [no data file]")
            continue

        df_raw = load_binance_json(path)
        df, X = MODEL_BUILDERS["C"](df_raw, vol_window=VOL_WINDOW)
        model = train_hmm(X, n_states=3)
        states = predict_states(model, X)
        df, _ = label_regimes(df, states)
        df = add_volume_surprise(df)

        means = df.groupby("regime")["volume_z"].mean()
        low  = means.get("Low (Bull)", np.nan)
        mid  = means.get("Medium (Sideways)", np.nan)
        high = means.get("High (Bear)", np.nan)
        mono = "yes" if low < mid < high else "no"
        print(f"{q:<4}{low:>12.3f}{mid:>12.3f}{high:>13.3f}{mono:>10}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "15m")
