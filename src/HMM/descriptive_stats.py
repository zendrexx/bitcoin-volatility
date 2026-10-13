import sys

from main import find_file
from data_loader import load_binance_json
from feature_engineering import MODEL_BUILDERS

COINS      = ["btc", "eth", "doge", "ltc"]
QUARTERS   = ["Q1", "Q2", "Q3", "Q4"]
VOL_WINDOW = 10
FEATURES   = ["log_return", "high_low_range", "volume", "rolling_vol"]


def run(tf: str = "15m", coins=COINS):
    for coin in coins:
        print(f"\n{coin.upper()} {tf} — raw feature characteristics by quarter (no HMM, no regime)")
        print(f"{'Q':<4}{'feature':<16}{'mean':>16}{'std':>16}{'skew':>8}{'kurt':>8}")

        for q in QUARTERS:
            path = find_file(tf, q, coin=coin)
            if path is None:
                print(f"{q:<4}  [no data file]")
                continue
            df_raw = load_binance_json(path)
            df, _ = MODEL_BUILDERS["A"](df_raw, vol_window=VOL_WINDOW)

            for feat in FEATURES:
                s = df[feat]
                print(f"{q:<4}{feat:<16}{s.mean():>16.6f}{s.std():>16.6f}{s.skew():>8.2f}{s.kurt():>8.2f}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "15m")
