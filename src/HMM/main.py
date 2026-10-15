"""Fit quarterly Models A-D for all supported coins and candle intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from data_loader import load_binance_json
from feature_engineering import MODEL_BUILDERS
from hmm_model import train_hmm, predict_states, compute_metrics, label_regimes, state_occupancy, is_degenerate
from export_to_json import export_json, export_results_summary
from regime_statistics import summarize_regimes

COINS = ["btc", "eth", "doge", "ltc"]
TIMEFRAMES = ["5m", "15m", "30m"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
YEAR = 2025
N_STATES = 3
VOL_WINDOW = 10
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUTPUT_DIR = DATA_DIR.parent / "processed"
MODEL_NAMES = ["A", "B", "C", "D"]


def find_file(tf: str, q: str, coin: str = "btc") -> str | None:
    if not DATA_DIR.is_dir():
        return None
    target = f"{coin}_{tf}_{q}_{YEAR}.json".lower()
    return next((str(p) for p in DATA_DIR.iterdir() if p.name.lower() == target), None)


def processed_label(coin, tf, quarter, model):
    # Preserve established BTC filenames; identify the other coins explicitly.
    prefix = "" if coin == "btc" else f"{coin}_"
    return f"{prefix}{tf}_{quarter}_Model{model}"


def result_key(result):
    return (result.get("coin", "BTC").lower(), result["timeframe"], result["quarter"], result["model"])


def run(coins=COINS, timeframes=TIMEFRAMES, quarters=QUARTERS):
    if set(coins) - set(COINS) or set(timeframes) - set(TIMEFRAMES) or set(quarters) - set(QUARTERS):
        raise ValueError("Unsupported coin, timeframe, or quarter")
    results = []
    for coin in coins:
        for tf in timeframes:
            for q in quarters:
                print(f"\n{coin.upper()} {tf} {q} {YEAR}", flush=True)
                path = find_file(tf, q, coin)
                raw = load_binance_json(path) if path else None
                assignments, quarter_results = {}, []
                for model_name in MODEL_NAMES:
                    builder = MODEL_BUILDERS[model_name]
                    label = processed_label(coin, tf, q, model_name)
                    result = {
                        "coin": coin.upper(), "training_year": YEAR, "timeframe": tf,
                        "quarter": q, "model": f"Model {model_name}",
                        "features": builder.__doc__.split("\n")[0].strip(),
                        "n_states": N_STATES,
                    }
                    try:
                        if raw is None:
                            raise FileNotFoundError(f"Missing {coin}_{tf}_{q}_{YEAR}.json")
                        df, X = builder(raw, vol_window=VOL_WINDOW)
                        if len(X) < N_STATES * 20:
                            raise ValueError(f"Not enough observations: {len(X)}")
                        model = train_hmm(X, n_states=N_STATES)
                        states = predict_states(model, X)
                        occupancy = state_occupancy(states, N_STATES)
                        if is_degenerate(states, N_STATES):
                            raise ValueError(f"Degenerate state occupancy: {occupancy.round(4).tolist()}")
                        df, mapping = label_regimes(df, states)
                        result.update(compute_metrics(model, X))
                        result.update(summarize_regimes(df, states, mapping, model.transmat_, tf))
                        output = export_json(df, label, output_dir=OUTPUT_DIR)
                        result.update({"status": "ok", "output_file": Path(output).name})
                        tier_map = {p["state"]: p["tier"] for p in result["state_profiles"]}
                        assignments[model_name] = np.array([tier_map[int(s)] for s in states])
                        print(f"  Model {model_name}: saved {len(df):,} candles", flush=True)
                    except Exception as exc:
                        result.update({"status": "error", "error": str(exc), "output_file": None})
                        stale = OUTPUT_DIR / f"hmm_{label}.json"
                        if stale.resolve().parent != OUTPUT_DIR.resolve():
                            raise ValueError("Processed output must stay within the output directory")
                        stale.unlink(missing_ok=True)
                        print(f"  Model {model_name}: ERROR {exc}", flush=True)
                    quarter_results.append(result)
                for result in quarter_results:
                    mid = result["model"][-1]
                    result["agreement_with_A"] = (
                        float(np.mean(assignments[mid] == assignments["A"]))
                        if mid in assignments and "A" in assignments else None
                    )
                results.extend(quarter_results)

    summary_path = OUTPUT_DIR / "hmm_results_summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    replacements = {result_key(r) for r in results}
    retained = [dict(r, coin=r.get("coin", "BTC")) for r in existing if result_key(r) not in replacements]
    combined = sorted(retained + results, key=result_key)
    export_results_summary(combined, output_dir=OUTPUT_DIR)
    failures = sum(r["status"] != "ok" for r in results)
    print(f"\nCompleted {len(results)} fits: {len(results)-failures} saved, {failures} failed.", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coin", choices=[*COINS, "all"], default="all")
    parser.add_argument("--tf", choices=[*TIMEFRAMES, "all"], default="all")
    parser.add_argument("--quarter", choices=[*QUARTERS, "all"], default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        results = run(
            COINS if args.coin == "all" else [args.coin],
            TIMEFRAMES if args.tf == "all" else [args.tf],
            QUARTERS if args.quarter == "all" else [args.quarter],
        )
    return int(any(r["status"] != "ok" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
