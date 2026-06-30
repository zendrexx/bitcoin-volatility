"""
server.py
Flask backend — reads your real btc_<tf>_<Q>_2025.json files,
runs the HMM for all 4 models, and returns results to the dashboard.

Now supports BTC, ETH, and DOGE — pass ?coin=eth or ?coin=doge to /api/run
and /api/files (defaults to btc if not specified).

Usage:
    pip install flask flask-cors hmmlearn scikit-learn pandas numpy
    python server.py

Then open: http://localhost:5050
Put your JSON files in data/raw/ relative to this script.
Naming: {coin}_{tf}_{quarter}_2025.json  e.g. eth_15m_Q1_2025.json
"""

import os, json, traceback
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
# If your data lives elsewhere relative to this script, override directly, e.g.:
# DATA_DIR = "src/HMM/data/raw"
TIMEFRAMES = ["5m", "15m", "30m"]
QUARTERS   = ["Q1", "Q2", "Q3", "Q4"]
YEAR       = 2025
VOL_WINDOW = 10

# Supported coins (short name used in filenames)
SUPPORTED_COINS = ["btc", "eth", "doge"]

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Data loading ──────────────────────────────────────────────────────────
KLINE_COLS = ["time","open","high","low","close","volume","close_time",
              "qav","trades","tbbav","tbqav","ignore"]
NUMERIC    = ["open","high","low","close","volume","qav","tbbav","tbqav"]

def find_file(coin: str, tf: str, quarter: str) -> str | None:
    """
    Case-insensitive file lookup. Looks for {coin}_{tf}_{quarter}_{YEAR}.json
    but tolerates different casing (ETH vs eth, q1 vs Q1, etc.) since that's
    a common source of "file not found" when files were renamed by hand.
    """
    if not os.path.isdir(DATA_DIR):
        return None
    target = f"{coin}_{tf}_{quarter}_{YEAR}.json".lower()
    for fname in os.listdir(DATA_DIR):
        if fname.lower() == target:
            return os.path.join(DATA_DIR, fname)
    return None

def load_file(coin: str, tf: str, quarter: str) -> pd.DataFrame | None:
    path = find_file(coin, tf, quarter)
    if path is None:
        return None
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data, columns=KLINE_COLS)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for c in NUMERIC:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df.sort_values("time").reset_index(drop=True)

# ── Feature builders ──────────────────────────────────────────────────────
def base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["rolling_vol"]    = df["log_return"].rolling(VOL_WINDOW).std()
    return df.dropna().reset_index(drop=True)

def scale(X):
    return StandardScaler().fit_transform(X)

MODELS = {
    "A": lambda df: scale(np.column_stack([df["log_return"], df["high_low_range"]])),
    "B": lambda df: scale(np.column_stack([df["log_return"], df["high_low_range"], df["volume"]])),
    "C": lambda df: scale(np.column_stack([df["log_return"], df["high_low_range"], df["rolling_vol"]])),
    "D": lambda df: scale(np.column_stack([df["log_return"], df["high_low_range"], df["volume"], df["rolling_vol"]])),
}

MODEL_FEATURES = {
    "A": "Log Return + High-Low Range",
    "B": "Log Return + High-Low Range + Volume",
    "C": "Log Return + High-Low Range + Rolling Volatility",
    "D": "Log Return + High-Low Range + Volume + Rolling Volatility",
}

# ── HMM metrics ───────────────────────────────────────────────────────────
def compute_metrics(model: GaussianHMM, X: np.ndarray) -> dict:
    n, n_feat = X.shape
    ns = model.n_components
    k  = ns*(ns-1) + (ns-1) + ns*n_feat + ns*n_feat*(n_feat+1)//2
    ll = model.score(X)
    return {
        "log_likelihood": round(float(ll), 4),
        "aic":            round(float(2*k - 2*ll), 4),
        "bic":            round(float(np.log(n)*k - 2*ll), 4),
        "n_params":       int(k),
        "n_obs":          int(n),
    }

# ── Core computation ──────────────────────────────────────────────────────
def run_hmm(coin: str, tf: str, quarter: str, n_states: int) -> list[dict]:
    df_raw = load_file(coin, tf, quarter)
    if df_raw is None:
        return []
    df = base_features(df_raw)
    if len(df) < n_states * 20:
        return []

    results = []
    for model_id, feature_fn in MODELS.items():
        try:
            X = feature_fn(df)
            hmm = GaussianHMM(n_components=n_states, covariance_type="full",
                              n_iter=200, random_state=42)
            hmm.fit(X)
            metrics = compute_metrics(hmm, X)

            # Regime labels ordered by mean rolling vol
            states = hmm.predict(X)
            df_tmp = df.copy()
            df_tmp["state"] = states
            vol_order = df_tmp.groupby("state")["rolling_vol"].mean().sort_values().index.tolist()
            n = len(vol_order)
            if n == 2:   names = ["Low Vol","High Vol"]
            elif n == 3: names = ["Bull","Sideways","Bear"]
            else:        names = [f"Regime {i}" for i in range(n)]
            state_map = {s: names[i] for i,s in enumerate(vol_order)}

            results.append({
                "coin":        coin.upper(),
                "timeframe":   tf,
                "quarter":     quarter,
                "model":       f"Model {model_id}",
                "features":    MODEL_FEATURES[model_id],
                "n_states":    n_states,
                **metrics,
                "state_map":   {str(k): v for k,v in state_map.items()},
                "transitions": hmm.transmat_.round(4).tolist(),
                "means":       hmm.means_.round(6).tolist(),
            })
        except Exception as e:
            results.append({
                "coin": coin.upper(), "timeframe": tf, "quarter": quarter,
                "model": f"Model {model_id}", "error": str(e)
            })
    return results

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/api/coins")
def api_coins():
    """List which coins have at least one data file present."""
    available = []
    for coin in SUPPORTED_COINS:
        for tf in TIMEFRAMES:
            for q in QUARTERS:
                if find_file(coin, tf, q) is not None:
                    available.append(coin)
                    break
            else:
                continue
            break
    return jsonify(available)

@app.route("/api/run")
def api_run():
    coin     = request.args.get("coin",   "btc").lower()
    tf       = request.args.get("tf",     "15m")
    n_states = int(request.args.get("states", 3))

    all_results = []
    available   = []
    missing     = []

    for q in QUARTERS:
        rows = run_hmm(coin, tf, q, n_states)
        if rows:
            all_results.extend(rows)
            available.append(q)
        else:
            missing.append(q)

    return jsonify({
        "results":   all_results,
        "available": available,
        "missing":   missing,
        "coin":      coin.upper(),
        "tf":        tf,
        "n_states":  n_states,
    })

@app.route("/api/files")
def api_files():
    """List which data files actually exist on disk for the given coin."""
    coin = request.args.get("coin", "btc").lower()
    found = {}
    for tf in TIMEFRAMES:
        found[tf] = []
        for q in QUARTERS:
            path = find_file(coin, tf, q)
            if path is not None:
                size = os.path.getsize(path)
                found[tf].append({"quarter": q, "size_mb": round(size/1e6, 2)})
    return jsonify(found)

@app.route("/api/debug")
def api_debug():
    """List every file actually present in DATA_DIR, for troubleshooting naming mismatches."""
    if not os.path.isdir(DATA_DIR):
        return jsonify({"data_dir": DATA_DIR, "exists": False, "files": []})
    files = sorted(os.listdir(DATA_DIR))
    return jsonify({"data_dir": DATA_DIR, "exists": True, "files": files})

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\nData directory: {DATA_DIR}")
    print("Supported coins: btc, eth, doge")
    print("Place your files as:  btc_5m_Q1_2025.json  (and eth_, doge_ variants, Q2/Q3/Q4, 15m, 30m)\n")
    app.run(port=5050, debug=True)