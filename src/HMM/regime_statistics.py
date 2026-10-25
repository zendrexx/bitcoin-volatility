"""Describe decoded volatility regimes in the original feature units."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ("log_return", "abs_log_return", "high_low_range", "volume", "rolling_vol")


def summarize_regimes(df, states, state_map, transitions, timeframe):
    """Summarize one fitted sequence, breaking observed runs at missing candles."""
    states = np.asarray(states, dtype=int)
    if len(df) != len(states) or not len(states):
        raise ValueError("State assignments must align with a nonempty feature frame")
    times = pd.to_datetime(df["time"])
    adjacent = times.diff().eq(pd.Timedelta(timeframe)).to_numpy()
    changed = np.r_[True, states[1:] != states[:-1]]
    starts = np.flatnonzero(changed | ~adjacent)
    lengths = np.diff(np.r_[starts, len(states)])
    run_states = states[starts]
    minutes = pd.Timedelta(timeframe).total_seconds() / 60
    profiles = []
    ordered_states = sorted(state_map, key=lambda s: df.loc[states == s, "rolling_vol"].mean())
    for tier, state in enumerate(ordered_states):
        sub = df.loc[states == state, list(FEATURE_COLUMNS)]
        runs = lengths[run_states == state]
        features = {}
        for column in FEATURE_COLUMNS:
            values = sub[column]
            features[column] = {
                "mean": float(values.mean()), "median": float(values.median()),
                "std": float(values.std()), "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            }
        profiles.append({
            "state": int(state), "tier": tier, "regime": state_map[state],
            "n_obs": len(sub), "occupancy": len(sub) / len(df),
            "mean_run_candles": float(runs.mean()),
            "median_run_candles": float(np.median(runs)),
            "median_run_minutes": float(np.median(runs) * minutes),
            "self_transition_probability": float(transitions[state, state]),
            "features": features,
        })
    return {
        "state_profiles": profiles,
        "switch_rate": float(changed[adjacent].mean()) if adjacent.any() else None,
        "transitions_by_regime": transitions[np.ix_(ordered_states, ordered_states)].tolist(),
    }
