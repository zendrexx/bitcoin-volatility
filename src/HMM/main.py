import os
import pandas as pd
import numpy as np

from data_loader import load_binance_json
from feature_engineering import MODEL_BUILDERS
from hmm_model import train_hmm, predict_states, compute_metrics, label_regimes, state_occupancy, is_degenerate
from export_to_json import export_json, export_results_summary

TIMEFRAMES   = ["5m", "15m", "30m"]
QUARTERS     = ["Q1", "Q2", "Q3", "Q4"]
YEAR         = 2025
N_STATES     = 3       # number of hidden states
VOL_WINDOW   = 10      # rolling volatility lookback (bars)
DATA_DIR     = "data/raw"
MODEL_NAMES  = ["A", "B", "C", "D"]


def find_file(tf: str, q: str) -> str | None:
    if not os.path.isdir(DATA_DIR):
        return None
    target = f"btc_{tf}_{q}_{YEAR}.json".lower()
    for fname in os.listdir(DATA_DIR):
        if fname.lower() == target:
            return os.path.join(DATA_DIR, fname)
    return None


def run():
    results = []

    for tf in TIMEFRAMES:
        for q in QUARTERS:
            file_path = find_file(tf, q)

            if file_path is None:
                print(f"[SKIP] btc_{tf}_{q}_{YEAR}.json not found in {DATA_DIR}.")
                continue

            print(f"\n{'='*50}")
            print(f"Timeframe: {tf}  |  Quarter: {q}")

            # Load raw data once
            df_raw = load_binance_json(file_path)

            for model_name in MODEL_NAMES:
                builder = MODEL_BUILDERS[model_name]

                # Build features
                df, X = builder(df_raw, vol_window=VOL_WINDOW)

                if len(X) < N_STATES * 10:
                    print(f"  [Model {model_name}] Not enough rows ({len(X)}), skipping.")
                    continue

                # Train
                try:
                    model = train_hmm(X, n_states=N_STATES)
                    states = predict_states(model, X)
                except Exception as e:
                    print(f"  [Model {model_name}] FAILED: {e}")
                    continue

                occ = state_occupancy(states, N_STATES)
                if is_degenerate(states, N_STATES):
                    print(
                        f"  [Model {model_name}] DEGENERATE: state occupancy "
                        f"{occ.round(4).tolist()} - a hidden state collapsed onto an "
                        f"outlier rather than a regime, skipping export."
                    )
                    continue

                # Metrics
                metrics = compute_metrics(model, X)

                # Label regimes
                df, state_map = label_regimes(df, states)

                # Export processed data
                label = f"{tf}_{q}_Model{model_name}"
                export_json(df, label)

                # Store result
                result = {
                    "timeframe":      tf,
                    "quarter":        q,
                    "model":          f"Model {model_name}",
                    "features":       builder.__doc__.split("\n")[0].strip(),
                    "log_likelihood": round(metrics["log_likelihood"], 4),
                    "aic":            round(metrics["aic"], 4),
                    "bic":            round(metrics["bic"], 4),
                    "n_params":       metrics["n_params"],
                    "n_obs":          metrics["n_obs"],
                    "n_states":       N_STATES,
                }
                results.append(result)

                print(
                    f"  Model {model_name} | "
                    f"LL: {metrics['log_likelihood']:.2f} | "
                    f"AIC: {metrics['aic']:.2f} | "
                    f"BIC: {metrics['bic']:.2f} | "
                    f"Rows: {metrics['n_obs']}"
                )

                # Per-state summary
                print(df.groupby("regime")[["log_return", "rolling_vol"]].mean().round(5).to_string())

    # Save summary
    summary_path = export_results_summary(results)
    print(f"\n\nResults saved to: {summary_path}")

    # Print final table
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    df_res = pd.DataFrame(results)
    print(df_res[["timeframe", "quarter", "model", "log_likelihood", "aic", "bic"]].to_string(index=False))

    return results


if __name__ == "__main__":
    run()