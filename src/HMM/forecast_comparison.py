"""Causal HMM forecasts and agreement with baseline A, in Low/Medium/High order.

No probability clipping or smoothing is applied. KL may be infinite. JSON
responses encode an infinite divergence as the string "Infinity".
"""

import math
import numpy as np
from scipy.special import logsumexp, rel_entr

STATE_ORDER = ["Low volatility", "Medium volatility", "High volatility"]
METRICS = ("overlap_percent", "jsd_bits", "kl_bits", "top1_agreement")


def normalize_probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not p.size or not np.all(np.isfinite(p)) or np.any(p < 0) or p.max() <= 0:
        raise ValueError("Expected a finite, nonnegative probability vector with positive mass")
    # Scale first to avoid overflow when normalizing finite, large weights.
    p = p / p.max()
    return p / p.sum()


def compare_distributions(base_probs, other_probs):
    p, q = normalize_probabilities(base_probs), normalize_probabilities(other_probs)
    if p.shape != q.shape:
        raise ValueError("Expected matching 1D probability vectors")
    midpoint = (p + q) / 2
    jsd = float((rel_entr(p, midpoint).sum() + rel_entr(q, midpoint).sum()) / (2 * np.log(2)))
    kl = float(rel_entr(p, q).sum() / np.log(2))
    if jsd < -1e-12 or jsd > 1 + 1e-12 or kl < -1e-12:
        raise ArithmeticError("Divergence outside its mathematical range")
    return {
        "overlap_percent": float(np.minimum(p, q).sum() * 100),
        "jsd_bits": min(1.0, max(0.0, jsd)),
        "kl_bits": max(0.0, kl),
        # Deterministic tie rule: first state in Low, Medium, High order.
        "top1_agreement": int(np.argmax(p) == np.argmax(q)),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        raise ValueError("NaN cannot be serialized as a forecast metric")
    return value


def aligned_comparisons(base_times, base_probs, other_times, other_probs):
    """Compare only exact matching origin timestamps; reject ambiguous duplicates."""
    if len(base_times) != len(base_probs) or len(other_times) != len(other_probs):
        raise ValueError("Forecast vectors must align with their timestamps")
    if len(set(base_times)) != len(base_times) or len(set(other_times)) != len(other_times):
        raise ValueError("Forecast timestamps must be unique")
    left, right = dict(zip(base_times, base_probs)), dict(zip(other_times, other_probs))
    return [{"timestamp": t, **compare_distributions(left[t], right[t])}
            for t in sorted(left.keys() & right.keys()) if left[t] is not None and right[t] is not None]


def summarize_comparisons(rows):
    """Include infinities in both mean and median, without capping or omission."""
    if not rows:
        return {"n_compared": 0, "infinite_kl_count": 0}
    kl = np.asarray([float(r["kl_bits"]) for r in rows])
    return {
        "n_compared": len(rows),
        "mean_overlap_percent": float(np.mean([r["overlap_percent"] for r in rows])),
        "mean_jsd_bits": float(np.mean([r["jsd_bits"] for r in rows])),
        "median_jsd_bits": float(np.median([r["jsd_bits"] for r in rows])),
        "mean_kl_bits": float(np.mean(kl)), "median_kl_bits": float(np.median(kl)),
        "infinite_kl_count": int(np.isinf(kl).sum()),
        "top1_agreement_percent": float(np.mean([r["top1_agreement"] for r in rows]) * 100),
    }


def filter_probabilities(hmm, X, previous=None):
    """Forward filtering P(S_t | X_1..X_t); no backward/smoothing pass."""
    emission = hmm._compute_log_likelihood(X)
    if np.isnan(emission).any() or np.isposinf(emission).any():
        raise ValueError("Invalid HMM emission log densities")
    transition = np.asarray(hmm.transmat_, dtype=float)
    if np.any(transition < 0) or not np.all(np.isfinite(transition)) or not np.allclose(transition.sum(axis=1), 1):
        raise ValueError("Invalid transition matrix")
    with np.errstate(divide="ignore"):
        log_transition = np.log(transition)
        log_previous = np.log(normalize_probabilities(previous)) if previous is not None else None
        log_start = np.log(normalize_probabilities(hmm.startprob_))
    filtered = []
    for log_density in emission:
        prior = log_start if log_previous is None else logsumexp(log_previous[:, None] + log_transition, axis=0)
        log_current = prior + log_density
        normalizer = logsumexp(log_current)
        if not np.isfinite(normalizer):
            raise ValueError("No state supports this observation")
        log_previous = log_current - normalizer
        filtered.append(np.exp(log_previous))
    return np.asarray(filtered)


def align_regime_probabilities(hidden_probs, state_tiers):
    values = np.asarray(hidden_probs, dtype=float)
    n = values.shape[-1]
    if set(state_tiers) != set(range(n)) or set(state_tiers.values()) != set(range(n)) or n != 3:
        raise ValueError("Expected a one-to-one hidden-state mapping to Low/Medium/High")
    aligned = np.empty_like(values)
    for hidden, tier in state_tiers.items():
        aligned[..., tier] = values[..., hidden]
    return aligned


def build_comparison_series(timestamps, probabilities, closed):
    pairs = {}
    for mid in "BCD":
        baseline, other = probabilities.get("A", []), probabilities.get(mid, [])
        columns = {metric: [] for metric in METRICS}
        for index in range(len(timestamps)):
            p = baseline[index] if index < len(baseline) else None
            q = other[index] if index < len(other) else None
            result = compare_distributions(p, q) if p is not None and q is not None else None
            for metric in METRICS:
                columns[metric].append(result[metric] if result else None)
        pairs[mid] = columns
    return json_safe({"timestamps": timestamps, "closed": closed,
                      "state_order": STATE_ORDER, "pairs": pairs})
