"""Refresh the guide's tables for every coin/interval without fitting models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from main import COINS, TIMEFRAMES, QUARTERS, OUTPUT_DIR
from regime_statistics import summarize_regimes
from server import load_file, base_features, MODEL_COLUMNS, MODEL_DIR

GUIDE = Path(__file__).resolve().parents[2] / "MODEL_GUIDE.md"
MARKER = "<!-- BEGIN COMPLETE MODEL TABLES -->"
NAMES = ["Low volatility", "Medium volatility", "High volatility"]


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def percent(value, digits=2):
    return f"{100 * value:.{digits}f}"


def triplet(profiles, getter, digits=4, factor=100):
    return " / ".join(f"{factor * getter(p):.{digits}f}" for p in profiles)


def annual_results():
    results = []
    for coin in COINS:
        for tf in TIMEFRAMES:
            frames = [load_file(coin, tf, q) for q in QUARTERS]
            if any(f is None for f in frames):
                raise ValueError(f"Missing raw data for {coin} {tf}")
            raw = pd.concat(frames, ignore_index=True).sort_values("time").drop_duplicates("time").reset_index(drop=True)
            frame = base_features(raw)
            assignments, selection = {}, []
            for mid, columns in MODEL_COLUMNS.items():
                bundle = joblib.load(Path(MODEL_DIR) / f"{coin}_{tf}_model_{mid}.joblib")
                if bundle["columns"] != columns or bundle["n_training_rows"] != len(frame):
                    raise ValueError(f"Outdated model: {coin} {tf} {mid}")
                states = bundle["hmm"].predict(bundle["scaler"].transform(frame[columns].to_numpy(dtype=float)))
                mapping = {s: NAMES[tier] for s, tier in bundle["state_tiers"].items()}
                summary = summarize_regimes(frame, states, mapping, bundle["hmm"].transmat_, tf)
                calculated = [p["features"]["rolling_vol"]["mean"] for p in summary["state_profiles"]]
                if not np.allclose(calculated, bundle["tier_mean_rolling_vol"], rtol=0, atol=1e-12):
                    raise ValueError(f"Stored state labels do not match training data: {coin} {tf} {mid}")
                assignments[mid] = np.array([bundle["state_tiers"][int(s)] for s in states])
                selection.append(dict(summary, coin=coin.upper(), timeframe=tf, model=f"Model {mid}", n_obs=len(frame)))
            for row in selection:
                row["agreement_with_A"] = float(np.mean(assignments[row["model"][-1]] == assignments["A"]))
            results.extend(selection)
    return results


def feature_rows(records, include_interval=False):
    rows = []
    for row in records:
        for profile in row["state_profiles"]:
            f = profile["features"]
            values = [row["model"][-1], profile["regime"].split()[0],
                      percent(f["abs_log_return"]["mean"], 4),
                      percent(f["high_low_range"]["mean"], 4),
                      f'{f["volume"]["mean"]:.2f}',
                      percent(f["rolling_vol"]["mean"], 4), percent(profile["occupancy"])]
            rows.append(([row["timeframe"]] if include_interval else []) + values)
    return rows


def quarterly_rows(records):
    rows = []
    for r in records:
        prefix = [r["quarter"], r["model"][-1]]
        if r.get("status") != "ok":
            rows.append(prefix + ["ERROR: " + r.get("error", "unknown fit failure").replace("|", "/").replace("\n", " ")] + ["gap"] * 4)
            continue
        p = r["state_profiles"]
        rows.append(prefix + [
            triplet(p, lambda v: v["features"]["abs_log_return"]["mean"]),
            triplet(p, lambda v: v["features"]["rolling_vol"]["mean"]),
            triplet(p, lambda v: v["occupancy"], digits=2),
            percent(r["switch_rate"]),
            percent(r["agreement_with_A"]) if r["agreement_with_A"] is not None else "unavailable",
        ])
    return rows


def replace_table(text, header, rows):
    start = text.index(header)
    end = text.index("\n\n", start)
    return text[:start] + markdown_table([v.strip() for v in header.strip("|").split("|")], rows) + text[end:]


def update_existing_tables(text, annual, quarterly):
    btc = [r for r in annual if r["coin"] == "BTC" and r["timeframe"] == "15m"]
    text = replace_table(text, "| Model | Low (%) | Medium (%) | High (%) |", [
        [r["model"][-1]] + [percent(p["features"]["rolling_vol"]["mean"], 4) for p in r["state_profiles"]] for r in btc
    ])
    baseline = [r for r in annual if r["model"] == "Model A"]
    text = replace_table(text, "| Coin | Interval | Low (%) | Medium (%) | High (%) |", [
        [r["coin"], r["timeframe"]] + [percent(p["features"]["rolling_vol"]["mean"], 4) for p in r["state_profiles"]] for r in baseline
    ])
    text = replace_table(text, "| Model | State | Mean absolute log return (%) | Mean range (%) | Mean volume (BTC) | Mean rolling volatility (%) | Share (%) |", feature_rows(btc))
    text = replace_table(text, "| Coin | Interval | A-B agreement (%) | A-C agreement (%) | A-D agreement (%) |", [
        [r["coin"], r["timeframe"]] + [percent(next(a["agreement_with_A"] for a in annual if a["coin"] == r["coin"] and a["timeframe"] == r["timeframe"] and a["model"] == f"Model {mid}")) for mid in "BCD"] for r in baseline
    ])
    example = [r for r in quarterly if r["coin"] == "BTC" and r["timeframe"] == "15m"]
    rows = []
    for r in example:
        if r.get("status") != "ok":
            message = r.get("error", "unknown fit failure").replace("|", "/").replace("\n", " ")
            rows.append([r["quarter"], r["model"][-1], "ERROR: " + message, "gap", "gap", "gap"])
            continue
        p = r["state_profiles"]
        rows.append([r["quarter"], r["model"][-1], triplet(p, lambda v:v["features"]["rolling_vol"]["mean"]), percent(p[2]["occupancy"]), percent(r["switch_rate"]), percent(r["agreement_with_A"])])
    return replace_table(text, "| Quarter | Model | Mean rolling volatility, L / M / H (%) | High share (%) | Switch rate (%) | Agreement with A (%) |", rows)


def render_complete_tables(annual, quarterly):
    lines = [MARKER, "## Complete annual regime profiles: every coin and interval", "",
        "These tables cover all 48 annual saved models and all 144 decoded states. Each coin has 5m, 15m, and 30m tables for Models A-D. Values are training descriptions in original units. Volume is the amount of the named coin traded per candle. Return magnitude, range, and rolling volatility are percentages. All models use the same candles within a coin and interval.", "",
        "Temporal summaries use observed consecutive runs, broken at missing candle intervals. Median run duration is in minutes. P(stay) is the learned self-transition probability for each ordered state, expressed as a percentage. Runs at the sample edges are truncated by the observation period.", ""]
    for coin in COINS:
        lines.extend([f"<details>\n<summary>{coin.upper()}: annual profiles at 5m, 15m, and 30m</summary>", ""])
        for tf in TIMEFRAMES:
            selection = [r for r in annual if r["coin"] == coin.upper() and r["timeframe"] == tf]
            lines.extend([f"### {coin.upper()} {tf}: annual fitted states", "", markdown_table(
                ["Model", "State", "Mean absolute return (%)", "Mean range (%)", f"Mean volume ({coin.upper()})", "Mean rolling volatility (%)", "Share (%)"], feature_rows(selection)), ""])
            temporal = [[r["model"][-1], triplet(r["state_profiles"], lambda p:p["median_run_minutes"], digits=1, factor=1), triplet(r["state_profiles"], lambda p:p["self_transition_probability"], digits=2), percent(r["switch_rate"])] for r in selection]
            lines.extend([markdown_table(["Model", "Median run L / M / H (minutes)", "P(stay) L / M / H (%)", "Switch rate (%)"], temporal), ""])
        lines.extend(["</details>", ""])
    successes = sum(r.get("status") == "ok" for r in quarterly)
    lines.extend(["## Complete quarterly comparison: every coin and interval", "",
        f"The batch attempted all 192 quarterly combinations: **{successes} successful fits and {len(quarterly)-successes} failures**. Each row below is one independent quarterly fit. Triples are Low / Medium / High. Shares and switching describe the retained sample; agreement describes similarity to A. These measures do not establish predictive accuracy or a universal winner.", "",
        "For every successful quarterly fit, [hmm_results_summary.json](src/HMM/data/processed/hmm_results_summary.json) also stores count, mean, median, standard deviation, first/third quartiles for all five feature columns, observed run lengths, and the transition matrix in regime order. Its `coin`, `timeframe`, `quarter`, and `model` fields identify each fit; `output_file` points to its candle-level JSON. Failed fits retain an explicit error and have no current candle output.", ""])
    for coin in COINS:
        lines.extend([f"<details>\n<summary>{coin.upper()}: quarterly results at 5m, 15m, and 30m</summary>", ""])
        for tf in TIMEFRAMES:
            selection = [r for r in quarterly if r["coin"] == coin.upper() and r["timeframe"] == tf]
            lines.extend([f"### {coin.upper()} {tf}: independent quarterly fits", "", markdown_table(
                ["Quarter", "Model", "Mean absolute return L / M / H (%)", "Mean rolling volatility L / M / H (%)", "Shares L / M / H (%)", "Switch rate (%)", "Agreement with A (%)"], quarterly_rows(selection)), ""])
        lines.extend(["</details>", ""])
    lines.append("<!-- END COMPLETE MODEL TABLES -->")
    return "\n".join(lines)


def main():
    quarterly = json.loads((OUTPUT_DIR / "hmm_results_summary.json").read_text(encoding="utf-8"))
    expected = {(c.upper(), tf, q, f"Model {m}") for c in COINS for tf in TIMEFRAMES for q in QUARTERS for m in "ABCD"}
    actual = {(r["coin"], r["timeframe"], r["quarter"], r["model"]) for r in quarterly}
    if len(quarterly) != 192 or expected != actual:
        raise ValueError("Run main.py for all coins and intervals before updating the guide")
    quarterly.sort(key=lambda r:(COINS.index(r["coin"].lower()), TIMEFRAMES.index(r["timeframe"]), r["quarter"], r["model"]))
    with threadpool_limits(limits=1):
        annual = annual_results()
    text = GUIDE.read_text(encoding="utf-8").split(MARKER)[0].rstrip()
    text = update_existing_tables(text, annual, quarterly)
    GUIDE.write_text(text + "\n\n" + render_complete_tables(annual, quarterly) + "\n", encoding="utf-8")
    print(f"Updated {GUIDE}: 48 annual models, 144 annual states, and 192 quarterly outcomes")


if __name__ == "__main__":
    main()
