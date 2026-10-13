import sys
import numpy as np

from main import find_file
from data_loader import load_binance_json
from feature_engineering import MODEL_BUILDERS
from hmm_model import train_hmm, predict_states, label_regimes, is_degenerate

COINS      = ["btc", "eth", "doge", "ltc"]
QUARTERS   = ["Q1", "Q2", "Q3", "Q4"]
VOL_WINDOW = 10
REGIMES    = ["Low (Bull)", "Medium (Sideways)", "High (Bear)"]


def run_lengths(states: np.ndarray, state_map: dict) -> dict:
    lengths = {name: [] for name in state_map.values()}
    run_state, run_len = states[0], 1
    for s in states[1:]:
        if s == run_state:
            run_len += 1
        else:
            lengths[state_map[run_state]].append(run_len)
            run_state, run_len = s, 1
    lengths[state_map[run_state]].append(run_len)
    return {name: np.mean(v) if v else np.nan for name, v in lengths.items()}


def run(tf: str = "15m", coins=COINS):
    for coin in coins:
        agg = {r: {"bars": 0, "log_return": 0.0, "high_low_range": 0.0,
                    "volume": 0.0, "rolling_vol": 0.0, "lengths": []} for r in REGIMES}
        total_bars = 0

        for q in QUARTERS:
            path = find_file(tf, q, coin=coin)
            if path is None:
                continue
            df_raw = load_binance_json(path)
            df, X = MODEL_BUILDERS["D"](df_raw, vol_window=VOL_WINDOW)
            model = train_hmm(X, n_states=3)
            states = predict_states(model, X)
            if is_degenerate(states, 3):
                print(f"{coin.upper()} {tf} {q}: DEGENERATE, skipped")
                continue
            df, state_map = label_regimes(df, states)
            lens = run_lengths(states, state_map)

            for r in REGIMES:
                sub = df[df["regime"] == r]
                n = len(sub)
                agg[r]["bars"] += n
                agg[r]["log_return"] += sub["log_return"].sum()
                agg[r]["high_low_range"] += sub["high_low_range"].sum()
                agg[r]["volume"] += sub["volume"].sum()
                agg[r]["rolling_vol"] += sub["rolling_vol"].sum()
                agg[r]["lengths"].append(lens[r])
            total_bars += len(df)

        print(f"\n{coin.upper()} {tf} — regime characteristics, Q1-Q4 pooled (Model D fit per quarter)")
        print(f"{'Regime':<20}{'% bars':>8}{'log_ret':>10}{'hl_range':>10}{'volume':>16}{'roll_vol(bp)':>13}{'avg_len':>9}")
        for r in REGIMES:
            n = agg[r]["bars"]
            if n == 0 or total_bars == 0:
                print(f"{r:<20}{'n/a':>8}")
                continue
            pct     = 100 * n / total_bars
            avg_ret = agg[r]["log_return"] / n
            avg_hl  = agg[r]["high_low_range"] / n
            avg_vol = agg[r]["volume"] / n
            avg_rv  = agg[r]["rolling_vol"] / n
            avg_len = np.nanmean(agg[r]["lengths"])
            print(f"{r:<20}{pct:>7.1f}%{avg_ret:>10.5f}{avg_hl:>10.5f}{avg_vol:>16.1f}{avg_rv*10000:>13.1f}{avg_len:>9.1f}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "15m")
