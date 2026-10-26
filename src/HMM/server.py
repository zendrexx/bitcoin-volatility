import os, json, time, tempfile, threading
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from forecast_comparison import (
    STATE_ORDER, filter_probabilities, align_regime_probabilities,
    normalize_probabilities, build_comparison_series,
)

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models", "2025")
BINANCE_REST_BASE = os.environ.get("BINANCE_REST_BASE", "https://data-api.binance.vision")

TIMEFRAMES = ["5m", "15m", "30m"]
QUARTERS   = ["Q1", "Q2", "Q3", "Q4"]
YEAR       = 2025
MODEL_VERSION = 2
VOL_WINDOW = 10


SUPPORTED_COINS = ["btc", "eth", "doge", "ltc"]
SYMBOLS = {coin: f"{coin.upper()}USDT" for coin in SUPPORTED_COINS}
MODEL_COLUMNS = {
    "A": ["abs_log_return", "high_low_range"],
    "B": ["abs_log_return", "high_low_range", "volume"],
    "C": ["abs_log_return", "high_low_range", "rolling_vol"],
    "D": ["abs_log_return", "high_low_range", "volume", "rolling_vol"],
}
MODEL_LOCKS = {(coin, tf): threading.Lock() for coin in SUPPORTED_COINS for tf in TIMEFRAMES}

app = Flask(__name__, static_folder=".")
CORS(app)

KLINE_COLS = ["time","open","high","low","close","volume","close_time",
              "qav","trades","tbbav","tbqav","ignore"]
NUMERIC    = ["open","high","low","close","volume","qav","tbbav","tbqav"]

def find_file(coin: str, tf: str, quarter: str) -> str | None:
   
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


def klines_to_dataframe(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=KLINE_COLS)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(None)
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["trades"] = pd.to_numeric(df["trades"], errors="raise").astype(int)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def epoch_milliseconds(values: pd.Series) -> list[int]:
    """Return UTC timestamps as milliseconds, independent of pandas' datetime unit."""
    times = pd.to_datetime(values, utc=True).dt.tz_convert(None)
    return times.astype("datetime64[ms]").astype("int64").tolist()


EXCLUDED_WINDOWS = [
    ("2025-10-10 20:00:00", "2025-10-11 04:00:00"),  # liquidation cascade; collapses hidden states under full covariance
]

MIN_STATE_OCCUPANCY = 0.01  # below this, a state has collapsed onto an outlier rather than modeling a regime

def drop_excluded(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exogenous-event bars *after* derived columns exist, so log_return
    and rolling_vol are never computed across the gap."""
    if "time" not in df.columns:
        return df
    keep = pd.Series(True, index=df.index)
    for start, end in EXCLUDED_WINDOWS:
        keep &= ~df["time"].between(pd.Timestamp(start), pd.Timestamp(end))
    return df[keep].reset_index(drop=True)

def base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["abs_log_return"] = df["log_return"].abs()
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["rolling_vol"]    = df["log_return"].rolling(VOL_WINDOW).std()
    df = df.dropna().reset_index(drop=True)
    return drop_excluded(df)

def scale(X):
    return StandardScaler().fit_transform(X)

MODELS = {
    "A": lambda df: scale(np.column_stack([df["abs_log_return"], df["high_low_range"]])),
    "B": lambda df: scale(np.column_stack([df["abs_log_return"], df["high_low_range"], df["volume"]])),
    "C": lambda df: scale(np.column_stack([df["abs_log_return"], df["high_low_range"], df["rolling_vol"]])),
    "D": lambda df: scale(np.column_stack([df["abs_log_return"], df["high_low_range"], df["volume"], df["rolling_vol"]])),
}

MODEL_FEATURES = {
    "A": "Absolute Log Return + High-Low Range",
    "B": "Absolute Log Return + High-Low Range + Volume",
    "C": "Absolute Log Return + High-Low Range + Rolling Volatility",
    "D": "Absolute Log Return + High-Low Range + Volume + Rolling Volatility",
}

def compute_metrics(model: GaussianHMM, X: np.ndarray) -> dict:
    n, n_feat = X.shape
    ns = model.n_components
    k_cov = {
        "spherical": ns,
        "diag":      ns * n_feat,
        "full":      ns * n_feat * (n_feat + 1) // 2,
        "tied":      n_feat * (n_feat + 1) // 2,
    }[model.covariance_type]
    k  = ns*(ns-1) + (ns-1) + ns*n_feat + k_cov
    ll = model.score(X)
    return {
        "log_likelihood": round(float(ll), 4),
        "aic":            round(float(2*k - 2*ll), 4),
        "bic":            round(float(np.log(n)*k - 2*ll), 4),
        "n_params":       int(k),
        "n_obs":          int(n),
    }

def run_hmm(coin: str, tf: str, quarter: str, n_states: int) -> tuple[list[dict], dict | None]:
    df_raw = load_file(coin, tf, quarter)
    if df_raw is None:
        return [], None
    df = base_features(df_raw)
    if len(df) < n_states * 20:
        return [], None

    results = []
    distribution_probabilities = {}
    regime_series = {
        "timestamps": epoch_milliseconds(df["time"]),
        "models": {},
        "price_timestamps": epoch_milliseconds(df_raw["time"]),
        "close_prices": df_raw["close"].astype(float).tolist(),
    }
    for model_id, feature_fn in MODELS.items():
        try:
            X = feature_fn(df)
            hmm = GaussianHMM(n_components=n_states, covariance_type="full",
                              n_iter=500, random_state=42)
            hmm.fit(X)
            states = hmm.predict(X)

            occupancy = np.bincount(states, minlength=n_states) / len(states)
            if occupancy.min() < MIN_STATE_OCCUPANCY:
                results.append({
                    "coin": coin.upper(), "timeframe": tf, "quarter": quarter,
                    "model": f"Model {model_id}",
                    "error": (
                        f"degenerate fit: state occupancy {occupancy.round(4).tolist()} "
                        f"(min {occupancy.min():.2%} < {MIN_STATE_OCCUPANCY:.0%}) - a hidden "
                        "state collapsed onto an outlier bar instead of modeling a regime"
                    ),
                    "degenerate": True,
                    "state_occupancy": occupancy.round(4).tolist(),
                })
                regime_series["models"][model_id] = [None] * len(states)
                distribution_probabilities[model_id] = [None] * len(df)
                continue

            metrics = compute_metrics(hmm, X)

            # Regime labels ordered by mean rolling vol
            df_tmp = df.copy()
            df_tmp["state"] = states
            mean_volatility = df_tmp.groupby("state")["rolling_vol"].mean().sort_values()
            vol_order = mean_volatility.index.tolist()
            n = len(vol_order)
            if n == 2:   names = ["Low volatility","High volatility"]
            elif n == 3: names = ["Low volatility","Medium volatility","High volatility"]
            else:        names = [f"Regime {i}" for i in range(n)]
            state_map = {s: names[i] for i,s in enumerate(vol_order)}
            regime_order = {state: tier for tier, state in enumerate(vol_order)}
            regime_characteristics = [
                {
                    "regime": names[tier],
                    "mean_rolling_vol": float(mean_volatility.loc[state]),
                    "n_candles": int(np.count_nonzero(states == state)),
                    "proportion": float(occupancy[state]),
                }
                for tier, state in enumerate(vol_order)
            ]
            regime_series["models"][model_id] = [
                int(regime_order[int(state)]) for state in states
            ]
            # Descriptive distributions conditional on the full-quarter fit.
            # These are not forecasts from parameters available earlier in 2025.
            if n_states == 3:
                filtered = filter_probabilities(hmm, X)
                next_states = align_regime_probabilities(filtered @ hmm.transmat_, regime_order)
                distribution_probabilities[model_id] = next_states.tolist()

            results.append({
                "coin":        coin.upper(),
                "timeframe":   tf,
                "quarter":     quarter,
                "model":       f"Model {model_id}",
                "features":    MODEL_FEATURES[model_id],
                "n_states":    n_states,
                **metrics,
                "state_map":   {str(k): v for k,v in state_map.items()},
                "regime_characteristics": regime_characteristics,
                "transitions": hmm.transmat_.round(4).tolist(),
                "means":       hmm.means_.round(6).tolist(),
            })
        except Exception as e:
            regime_series["models"][model_id] = [None] * len(df)
            distribution_probabilities[model_id] = [None] * len(df)
            results.append({
                "coin": coin.upper(), "timeframe": tf, "quarter": quarter,
                "model": f"Model {model_id}", "error": str(e)
            })
    if n_states == 3:
        regime_series["distribution_comparison"] = build_comparison_series(
            regime_series["timestamps"], distribution_probabilities, [True] * len(df))
        regime_series["distribution_comparison"].update({
            "method": "in_sample_quarter_fit",
            "is_out_of_sample": False,
            "past_only_forecasts_available": False,
            "description": "Descriptive next-state distributions conditional on full-quarter fitted parameters",
        })
    return results, regime_series


def saved_model_path(coin: str, tf: str, model_id: str) -> str:
    return os.path.join(MODEL_DIR, f"{coin}_{tf}_model_{model_id}.joblib")


def train_2025_models(coin: str, tf: str, force: bool = False) -> list[dict]:
    """Fit one year per feature set and persist the fitted HMM and its scaler."""
    if coin not in SUPPORTED_COINS or tf not in TIMEFRAMES:
        raise ValueError("Unsupported coin or timeframe")
    with MODEL_LOCKS[(coin, tf)]:
        valid_saved = {} if force else load_saved_models(coin, tf)[0]
        if len(valid_saved) == len(MODEL_COLUMNS):
            return [{"model": model_id, "status": "saved"}
                    for model_id in MODEL_COLUMNS]
        missing_quarters = [q for q in QUARTERS if find_file(coin, tf, q) is None]
        if missing_quarters:
            raise FileNotFoundError(
                f"Missing 2025 training files: {', '.join(missing_quarters)}"
            )
        frames = [load_file(coin, tf, q) for q in QUARTERS]
        raw = pd.concat(frames, ignore_index=True).sort_values("time")
        raw = raw.drop_duplicates("time", keep="last").reset_index(drop=True)
        training = base_features(raw)
        if len(training) < 100:
            raise ValueError("Not enough 2025 observations to train")

        os.makedirs(MODEL_DIR, exist_ok=True)
        trained = []
        for model_id, columns in MODEL_COLUMNS.items():
            path = saved_model_path(coin, tf, model_id)
            if model_id in valid_saved:
                trained.append({"model": model_id, "status": "saved"})
                continue
            try:
                scaler = StandardScaler()
                X = scaler.fit_transform(training[columns].to_numpy(dtype=float))
                accepted = None
                for seed in (42, 7, 2025):
                    hmm = GaussianHMM(
                        n_components=3, covariance_type="full",
                        n_iter=100, tol=0.01, random_state=seed,
                    )
                    hmm.fit(X)
                    states = hmm.predict(X)
                    occupancy = np.bincount(states, minlength=3) / len(states)
                    if occupancy.min() >= MIN_STATE_OCCUPANCY:
                        accepted = (hmm, states, occupancy)
                        break
                if accepted is None:
                    raise ValueError("All 2025 fits had a collapsed volatility state")
                hmm, states, occupancy = accepted
                mean_volatility = training.assign(state=states).groupby("state")[
                    "rolling_vol"
                ].mean().sort_values()
                tiers = {int(state): int(tier) for tier, state in enumerate(mean_volatility.index)}
                bundle = {
                    "version": MODEL_VERSION, "training_year": YEAR, "coin": coin, "tf": tf,
                    "model_id": model_id, "columns": columns,
                    "scaler": scaler, "hmm": hmm, "state_tiers": tiers,
                    "tier_mean_rolling_vol": mean_volatility.astype(float).tolist(),
                    "n_training_rows": len(training),
                    "training_start_ms": int(epoch_milliseconds(training["time"].iloc[:1])[0]),
                    "training_end_ms": int(epoch_milliseconds(training["time"].iloc[-1:])[0]),
                    "state_occupancy": occupancy.tolist(),
                }
                with tempfile.NamedTemporaryFile(
                    dir=MODEL_DIR, suffix=".joblib", delete=False
                ) as temporary:
                    temporary_path = temporary.name
                try:
                    joblib.dump(bundle, temporary_path)
                    os.replace(temporary_path, path)
                finally:
                    if os.path.exists(temporary_path):
                        os.remove(temporary_path)
                trained.append({"model": model_id, "status": "trained",
                                "rows": len(training)})
            except Exception as exc:
                trained.append({"model": model_id, "status": "error", "error": str(exc)})
        return trained


@lru_cache(maxsize=96)
def _read_saved_model(path: str, modified_ns: int) -> dict:
    return joblib.load(path)


def load_saved_models(coin: str, tf: str) -> tuple[dict, list[str]]:
    bundles, missing = {}, []
    for model_id in MODEL_COLUMNS:
        path = saved_model_path(coin, tf, model_id)
        if not os.path.exists(path):
            missing.append(model_id)
            continue
        try:
            bundle = _read_saved_model(path, os.stat(path).st_mtime_ns)
        except Exception:
            missing.append(model_id)
            continue
        if not isinstance(bundle, dict):
            missing.append(model_id)
            continue
        if (bundle.get("version"), bundle.get("training_year"),
                bundle.get("coin"), bundle.get("tf"), bundle.get("model_id")) != (
                MODEL_VERSION, YEAR, coin, tf, model_id):
            missing.append(model_id)
            continue
        if bundle.get("columns") != MODEL_COLUMNS[model_id]:
            missing.append(model_id)
            continue
        bundles[model_id] = bundle
    return bundles, missing


def fetch_recent_klines(coin: str, tf: str, days: int = 30) -> list:
    interval_ms = int(tf[:-1]) * 60_000
    now_ms = int(time.time() * 1000)
    next_ms = now_ms - days * 86_400_000 - (VOL_WINDOW + 2) * interval_ms
    rows = []
    with requests.Session() as session:
        while next_ms <= now_ms:
            response = session.get(
                f"{BINANCE_REST_BASE}/api/v3/klines",
                params={"symbol": SYMBOLS[coin], "interval": tf,
                        "startTime": next_ms, "endTime": now_ms, "limit": 1000},
                timeout=15,
            )
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                raise ValueError("Unexpected Binance kline response")
            if not batch:
                break
            rows.extend(batch)
            following = int(batch[-1][0]) + interval_ms
            if following <= next_ms:
                raise ValueError("Binance kline pagination did not advance")
            next_ms = following
            if len(batch) < 1000:
                break
    return rows


def infer_saved_models(df_raw: pd.DataFrame, bundles: dict, filter_state=None, as_of_ms=None) -> dict:
    features = base_features(df_raw)
    if len(features) < 3:
        raise ValueError("Not enough recent candles for rolling volatility")
    as_of_ms = int(time.time() * 1000) if as_of_ms is None else int(as_of_ms)
    tf = next(iter(bundles.values()))["tf"] if bundles else "15m"
    interval_ms = int(tf[:-1]) * 60_000
    times = np.asarray(epoch_milliseconds(features["time"]), dtype=np.int64)
    if times[-1] > as_of_ms:
        raise ValueError("A candle origin cannot be in the future")
    seed_time = int(filter_state["time"]) if filter_state else None
    if seed_time is not None and seed_time + interval_ms - 1 >= as_of_ms:
        raise ValueError("Filter context must come from a closed candle")
    selected = np.flatnonzero(times > seed_time) if seed_time is not None else np.arange(len(times))
    reuse_seed = seed_time is not None and not len(selected) and seed_time == times[-1]
    if reuse_seed:
        selected = np.array([len(times) - 1])
    if not len(selected) or (seed_time is not None and not reuse_seed and times[selected[0]] != seed_time + interval_ms):
        raise ValueError("Filter context is outside the available candle history; refresh live data")
    selected_times = times[selected].tolist()
    closed = [t + interval_ms - 1 < as_of_ms for t in selected_times]
    series = {
        "price_timestamps": epoch_milliseconds(df_raw["time"]),
        "close_prices": df_raw["close"].astype(float).tolist(),
        "timestamps": selected_times,
        "models": {},
        "forecast": {},
        "filter_state": None,
    }
    next_probabilities, saved_posteriors = {}, {}
    for model_id in MODEL_COLUMNS:
        if model_id not in bundles:
            series["models"][model_id] = [None] * len(selected)
            series["forecast"][model_id] = None
            next_probabilities[model_id] = [None] * len(selected)
            continue
        bundle = bundles[model_id]
        hmm = bundle["hmm"]
        signature = joblib.hash((bundle["columns"], bundle["state_tiers"],
            hmm.startprob_, hmm.transmat_, hmm.means_, hmm.covars_,
            bundle["scaler"].mean_, bundle["scaler"].scale_))
        previous = None
        if filter_state:
            saved = filter_state.get("models", {}).get(model_id)
            if not saved or saved.get("signature") != signature:
                raise ValueError("Saved models changed or filter context is missing; refresh live data")
            previous = normalize_probabilities(saved["posterior"])
            if len(previous) != hmm.n_components:
                raise ValueError("Filter context has the wrong hidden-state dimension")
        X = bundle["scaler"].transform(
            features.iloc[selected][bundle["columns"]].to_numpy(dtype=float)
        )
        posterior = previous[None, :] if reuse_seed else filter_probabilities(hmm, X, previous)
        states = posterior.argmax(axis=1)
        series["models"][model_id] = [
            bundle["state_tiers"][int(state)] for state in states
        ]
        next_tiers = align_regime_probabilities(posterior @ hmm.transmat_, bundle["state_tiers"])
        cutoff = int(bundle["training_end_ms"])
        next_probabilities[model_id] = [p.tolist() if t > cutoff else None for t, p in zip(selected_times, next_tiers)]
        series["forecast"][model_id] = {
            "current_state": bundle["state_tiers"][int(states[-1])],
            "current_probability": float(posterior[-1, states[-1]]),
            "next_state_probabilities": next_probabilities[model_id][-1],
            "tier_mean_rolling_vol": bundle["tier_mean_rolling_vol"],
            "origin_time": selected_times[-1], "target_time": selected_times[-1] + interval_ms,
            "provisional": not closed[-1],
        }
        if any(closed):
            last_closed = max(i for i, value in enumerate(closed) if value)
            saved_posteriors[model_id] = {"posterior": posterior[last_closed].tolist(), "signature": signature}
    if saved_posteriors:
        series["filter_state"] = {"time": selected_times[last_closed], "models": saved_posteriors}
    series["forecast_comparison"] = build_comparison_series(selected_times, next_probabilities, closed)
    series["forecast_comparison"].update({
        "method": "causal_filter", "target_timestamps": [t + interval_ms for t in selected_times],
        "information_timestamps": [min(t + interval_ms - 1, as_of_ms) for t in selected_times],
        "trained_through": {mid: int(b["training_end_ms"]) for mid, b in bundles.items()},
    })
    for mid in "BCD":
        if series["forecast"].get(mid):
            columns = series["forecast_comparison"]["pairs"][mid]
            series["forecast"][mid]["comparison_with_A"] = {key: values[-1] for key, values in columns.items()}
    return series

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/api/coins")
def api_coins():
   
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
    regime_series = {}
    available   = []
    missing     = []

    for q in QUARTERS:
        rows, series = run_hmm(coin, tf, q, n_states)
        if rows:
            all_results.extend(rows)
            available.append(q)
        else:
            missing.append(q)
        if series is not None:
            regime_series[q] = series

    return jsonify({
        "results":   all_results,
        "regime_series": regime_series,
        "available": available,
        "missing":   missing,
        "coin":      coin.upper(),
        "tf":        tf,
        "n_states":  n_states,
    })


@app.route("/api/models/status")
def api_model_status():
    coin = request.args.get("coin", "btc").lower()
    tf = request.args.get("tf", "15m")
    if coin not in SUPPORTED_COINS or tf not in TIMEFRAMES:
        return jsonify({"error": "Unsupported coin or timeframe"}), 400
    bundles, missing = load_saved_models(coin, tf)
    return jsonify({
        "coin": coin.upper(), "tf": tf, "training_year": YEAR,
        "saved": list(bundles), "missing": missing,
    })


@app.route("/api/live")
def api_live():
    coin = request.args.get("coin", "btc").lower()
    tf = request.args.get("tf", "15m")
    if coin not in SUPPORTED_COINS or tf not in TIMEFRAMES:
        return jsonify({"error": "Unsupported coin or timeframe"}), 400
    try:
        bundles, missing = load_saved_models(coin, tf)
        training = train_2025_models(coin, tf) if missing else []
        bundles, missing = load_saved_models(coin, tf)
        if not bundles:
            return jsonify({"error": "No valid 2025 models could be trained",
                            "training": training}), 409
        rows = fetch_recent_klines(coin, tf)
        if not rows:
            raise ValueError("Binance returned no recent candles")
        df_raw = klines_to_dataframe(rows)
        if df_raw["time"].iloc[-1] < pd.Timestamp(f"{YEAR + 1}-01-01"):
            raise ValueError("Live candles are not later than the 2025 training set")
        now_ms = int(time.time() * 1000)
        interval_ms = int(tf[:-1]) * 60_000
        if now_ms - int(rows[-1][0]) > 2 * interval_ms:
            raise ValueError("The latest Binance candle is stale")
        series = infer_saved_models(df_raw, bundles, as_of_ms=now_ms)
        return jsonify({
            "coin": coin.upper(), "tf": tf, "source": "Binance Spot",
            "training_year": YEAR, "as_of": now_ms,
            "regime_series": {"Q1": series},
            "bars": rows[-300:],
            "provisional": int(rows[-1][6]) >= now_ms,
            "missing_models": missing,
            "training": training,
        })
    except (requests.RequestException, ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/live/infer", methods=["POST"])
def api_live_infer():
    payload = request.get_json(silent=True) or {}
    coin = str(payload.get("coin", "")).lower()
    tf = str(payload.get("tf", ""))
    rows = payload.get("bars", [])
    if coin not in SUPPORTED_COINS or tf not in TIMEFRAMES:
        return jsonify({"error": "Unsupported coin or timeframe"}), 400
    if not isinstance(rows, list) or not 12 <= len(rows) <= 300:
        return jsonify({"error": "Expected 12–300 recent candles"}), 400
    try:
        df_raw = klines_to_dataframe(rows)
        if df_raw["time"].iloc[-1] < pd.Timestamp(f"{YEAR + 1}-01-01"):
            return jsonify({"error": "Inference requires post-2025 candles"}), 400
        bundles, missing = load_saved_models(coin, tf)
        if not bundles:
            return jsonify({"error": "No saved 2025 models"}), 409
        series = infer_saved_models(df_raw, bundles, filter_state=payload.get("filter_state"))
        return jsonify({
            "time": series["price_timestamps"][-1],
            "close": series["close_prices"][-1],
            "models": {model_id: values[-1] for model_id, values in series["models"].items()},
            "forecast": series["forecast"],
            "timestamps": series["timestamps"], "model_series": series["models"],
            "forecast_comparison": series["forecast_comparison"],
            "filter_state": series["filter_state"],
            "missing_models": missing,
            "provisional": not bool(payload.get("closed", False)),
        })
    except (ValueError, KeyError, TypeError) as exc:
        if payload.get("filter_state"):
            return jsonify({"error": str(exc), "reset_required": True}), 409
        return jsonify({"error": str(exc)}), 400

@app.route("/api/characteristics")
def api_characteristics():
    """Raw descriptive stats per quarter — no HMM, no regime. Answers RQ1/RQ3 directly."""
    coin = request.args.get("coin", "btc").lower()
    tf   = request.args.get("tf", "15m")

    quarters = {}
    for q in QUARTERS:
        df_raw = load_file(coin, tf, q)
        if df_raw is None:
            continue
        df = base_features(df_raw)
        quarters[q] = {
            feat: {
                "mean": round(float(df[feat].mean()), 6),
                "std":  round(float(df[feat].std()), 6),
                "skew": round(float(df[feat].skew()), 2),
                "kurt": round(float(df[feat].kurt()), 2),
            }
            for feat in ["log_return", "high_low_range", "volume", "rolling_vol"]
        }
    return jsonify({"coin": coin.upper(), "tf": tf, "quarters": quarters})

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
    if not os.path.isdir(DATA_DIR):
        return jsonify({"data_dir": DATA_DIR, "exists": False, "files": []})
    files = sorted(os.listdir(DATA_DIR))
    return jsonify({"data_dir": DATA_DIR, "exists": True, "files": files})

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\nData directory: {DATA_DIR}")
    print("Supported coins: btc, eth, doge, ltc")
    print("Place your files as:  btc_5m_Q1_2025.json  (and eth_, doge_, ltc_ variants, Q2/Q3/Q4, 15m, 30m)\n")
    app.run(port=5050, debug=True)
