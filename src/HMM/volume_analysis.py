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
