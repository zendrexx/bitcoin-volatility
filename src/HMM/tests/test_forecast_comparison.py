import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from forecast_comparison import (
    compare_distributions, aligned_comparisons, summarize_comparisons,
    filter_probabilities, align_regime_probabilities, json_safe, STATE_ORDER,
)


def make_hmm(dimensions=2):
    model = GaussianHMM(3, covariance_type="full")
    model.n_features = dimensions
    model.startprob_ = np.array([0.6, 0.3, 0.1])
    model.transmat_ = np.array([[0.85, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    model.means_ = np.repeat(np.array([[0.0], [0.5], [2.0]]), dimensions, axis=1)
    model.covars_ = np.repeat(np.eye(dimensions)[None], 3, axis=0)
    return model


def make_bundles():
    bundles = {}
    for mid, columns in server.MODEL_COLUMNS.items():
        scaler = StandardScaler()
        scaler.mean_ = np.array([100.0 if c == "volume" else 0.0 for c in columns])
        scaler.scale_ = np.array([50.0 if c == "volume" else 0.01 for c in columns])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(columns)
        bundles[mid] = {"tf": "15m", "columns": columns, "scaler": scaler,
                        "hmm": make_hmm(len(columns)), "state_tiers": {0: 0, 1: 1, 2: 2},
                        "training_end_ms": int(pd.Timestamp("2025-12-31").timestamp() * 1000),
                        "tier_mean_rolling_vol": [0.001, 0.002, 0.004]}
    return bundles


def candles(n=80):
    times = pd.date_range("2026-01-10", periods=n, freq="15min")
    close = 100 * np.exp(np.cumsum(np.sin(np.arange(n)) * 0.001))
    return pd.DataFrame({"time": times, "close": close, "open": close,
                         "high": close + 0.2, "low": close - 0.2,
                         "volume": 100 + np.arange(n),
                         "close_time": times.astype("datetime64[ms]").astype("int64") + 899999})


def raw_rows(frame):
    return [[int(pd.Timestamp(r.time).timestamp() * 1000), r.open, r.high, r.low, r.close,
             r.volume, int(r.close_time), r.volume * r.close, 10, r.volume / 2, 1, 0]
            for r in frame.itertuples()]


class DistributionTests(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(compare_distributions([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
                         {"overlap_percent": 100.0, "jsd_bits": 0.0, "kl_bits": 0.0, "top1_agreement": 1})

    def test_jsd_symmetric_but_kl_directional(self):
        p, q = [0.8, 0.15, 0.05], [0.2, 0.3, 0.5]
        a, b = compare_distributions(p, q), compare_distributions(q, p)
        self.assertAlmostEqual(a["jsd_bits"], b["jsd_bits"])
        self.assertNotAlmostEqual(a["kl_bits"], b["kl_bits"])
        self.assertAlmostEqual(a["jsd_bits"], jensenshannon(p, q, base=2) ** 2)

    def test_random_bounds_and_overlap_equivalence(self):
        rng = np.random.default_rng(13)
        for _ in range(150):
            p, q = rng.dirichlet([0.5] * 3, size=2)
            result = compare_distributions(p, q)
            self.assertGreaterEqual(result["jsd_bits"], 0)
            self.assertLessEqual(result["jsd_bits"], 1)
            self.assertGreaterEqual(result["kl_bits"], 0)
            self.assertAlmostEqual(result["overlap_percent"], (1 - np.abs(p-q).sum()/2)*100)

    def test_zeros_infinite_kl_and_disjoint_jsd(self):
        result = compare_distributions([1, 0, 0], [0, 1, 0])
        self.assertEqual(result["overlap_percent"], 0)
        self.assertEqual(result["jsd_bits"], 1)
        self.assertTrue(math.isinf(result["kl_bits"]))
        self.assertEqual(compare_distributions([0, 1, 0], [0, 1, 0])["kl_bits"], 0)

    def test_invalid_vectors(self):
        for vector in ([], [0, 0, 0], [-1, 1, 1], [math.nan, 1, 0], [math.inf, 0, 0], [[1, 0, 0]]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                compare_distributions(vector, [1, 0, 0])
        with self.assertRaises(ValueError):
            compare_distributions([1, 0], [1, 0, 0])

    def test_normalization_and_tie_order(self):
        self.assertEqual(compare_distributions([2, 2, 0], [1, 1, 0])["jsd_bits"], 0)
        self.assertEqual(compare_distributions([1e308, 1e308, 0], [0.5, 0.5, 0])["jsd_bits"], 0)
        self.assertEqual(compare_distributions([0.5, 0.5, 0], [0.4, 0.6, 0])["top1_agreement"], 0)

    def test_summary_keeps_infinities_in_mean_and_median(self):
        finite = compare_distributions([0.5, 0.5, 0], [0.5, 0.5, 0])
        infinite = compare_distributions([1, 0, 0], [0, 1, 0])
        summary = summarize_comparisons([finite, finite, infinite])
        self.assertEqual(summary["infinite_kl_count"], 1)
        self.assertTrue(math.isinf(summary["mean_kl_bits"]))
        self.assertEqual(summary["median_kl_bits"], 0)
        self.assertTrue(math.isinf(summarize_comparisons([finite, infinite])["median_kl_bits"]))
        encoded = json.dumps(json_safe(summary), allow_nan=False)
        self.assertEqual(json.loads(encoded)["mean_kl_bits"], "Infinity")

    def test_timestamp_intersection_not_array_position(self):
        result = aligned_comparisons([20, 0, 10], [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                                     [30, 10, 0], [[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        self.assertEqual([r["timestamp"] for r in result], [0, 10])
        self.assertTrue(all(r["overlap_percent"] == 100 for r in result))
        with self.assertRaises(ValueError):
            aligned_comparisons([0, 0], [[1, 0, 0]] * 2, [0], [[1, 0, 0]])


class CausalForecastTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.bundles = candles(), make_bundles()

    def infer(self, frame=None, **kwargs):
        frame = self.frame if frame is None else frame
        return server.infer_saved_models(frame, self.bundles,
            as_of_ms=int(frame.close_time.iloc[-1]) + 1000, **kwargs)

    def test_prefix_invariant_and_future_data_cannot_change_past(self):
        prefix = self.infer(self.frame.iloc[:40])
        full = self.infer()
        future = self.frame.copy()
        future.loc[40:, ["close", "high", "volume"]] *= 3
        changed = self.infer(future)
        for mid in "BCD":
            for key in ("overlap_percent", "jsd_bits", "kl_bits", "top1_agreement"):
                values = prefix["forecast_comparison"]["pairs"][mid][key]
                np.testing.assert_allclose(values, full["forecast_comparison"]["pairs"][mid][key][:len(values)])
                np.testing.assert_allclose(values, changed["forecast_comparison"]["pairs"][mid][key][:len(values)])
        for origin, available, target in zip(full["timestamps"], full["forecast_comparison"]["information_timestamps"], full["forecast_comparison"]["target_timestamps"]):
            self.assertEqual(available, origin + 899999)
            self.assertLess(available, target)

    def test_hidden_state_permutations_preserve_aligned_forecasts(self):
        model, permutation = make_hmm(), [2, 0, 1]
        other = copy.deepcopy(model)
        other.startprob_ = model.startprob_[permutation]
        other.transmat_ = model.transmat_[np.ix_(permutation, permutation)]
        other.means_, other.covars_ = model.means_[permutation], model.covars_[permutation]
        X = np.array([[0, 0], [1, 1], [3, 2], [1, 2]], dtype=float)
        first = align_regime_probabilities(filter_probabilities(model, X) @ model.transmat_, {0: 0, 1: 1, 2: 2})
        second = align_regime_probabilities(filter_probabilities(other, X) @ other.transmat_, {i: permutation[i] for i in range(3)})
        np.testing.assert_allclose(first, second, atol=1e-14)
        for p, q in zip(first, second):
            self.assertAlmostEqual(compare_distributions(p, q)["jsd_bits"], 0)

    def test_last_filtered_probability_matches_prefix_posterior(self):
        model, X = make_hmm(), np.array([[0, 0], [1, 1], [3, 2], [1, 2]], dtype=float)
        filtered = filter_probabilities(model, X)
        for end in range(1, len(X)+1):
            np.testing.assert_allclose(filtered[end-1], model.predict_proba(X[:end])[-1], atol=1e-13)

    def test_streaming_context_matches_full_history_and_open_repeats(self):
        prefix = self.frame.iloc[:40]
        initial = server.infer_saved_models(prefix, self.bundles, as_of_ms=int(prefix.close_time.iloc[-1])-300000)
        context = initial["filter_state"]
        self.assertEqual(context["time"], initial["timestamps"][-2])
        for _ in range(2):
            updated = server.infer_saved_models(prefix, self.bundles, filter_state=context,
                                               as_of_ms=int(prefix.close_time.iloc[-1])-300000)
            for mid in "ABCD":
                np.testing.assert_allclose(updated["forecast"][mid]["next_state_probabilities"], initial["forecast"][mid]["next_state_probabilities"])
        streamed, full = self.infer(self.frame.iloc[:41], filter_state=context), self.infer(self.frame.iloc[:41])
        for mid in "ABCD":
            np.testing.assert_allclose(streamed["forecast"][mid]["next_state_probabilities"], full["forecast"][mid]["next_state_probabilities"], atol=1e-14)

    def test_training_cutoff_blocks_in_sample_forecast_comparisons(self):
        for bundle in self.bundles.values():
            bundle["training_end_ms"] = int(self.frame.close_time.iloc[-1])
        result = self.infer()
        for mid in "BCD":
            self.assertTrue(all(v is None for v in result["forecast_comparison"]["pairs"][mid]["jsd_bits"]))

    def test_missing_baseline_and_state_order(self):
        del self.bundles["A"]
        result = self.infer()
        self.assertEqual(result["forecast_comparison"]["state_order"], STATE_ORDER)
        for mid in "BCD":
            self.assertTrue(all(v is None for v in result["forecast_comparison"]["pairs"][mid]["kl_bits"]))

    def test_api_response_does_not_fit_any_model(self):
        rows = raw_rows(self.frame)
        with patch.object(server, "load_saved_models", return_value=(self.bundles, [])), \
             patch.object(GaussianHMM, "fit", side_effect=AssertionError("No retraining allowed")):
            response = server.app.test_client().post("/api/live/infer", json={"coin": "btc", "tf": "15m", "bars": rows, "closed": True})
        self.assertEqual(response.status_code, 200, response.json)
        result = response.json
        self.assertEqual(result["timestamps"], result["forecast_comparison"]["timestamps"])
        self.assertEqual(set(result["forecast_comparison"]["pairs"]), set("BCD"))
        json.dumps(result, allow_nan=False)


class QuarterlyDistributionTests(unittest.TestCase):
    def quarterly_result(self, degenerate_model=None):
        models = []

        def factory(**kwargs):
            mid = "ABCD"[len(models)]
            model = make_hmm()

            def fit(X):
                # Deterministic fixed parameters, with no real model training.
                fitted = make_hmm(X.shape[1])
                model.n_features = fitted.n_features
                model.means_, model.covars_ = fitted.means_, fitted.covars_
                return model

            model.fit = Mock(side_effect=fit)
            model.predict = Mock(side_effect=lambda X: np.zeros(len(X), dtype=int)
                                 if mid == degenerate_model else np.arange(len(X)) % 3)
            models.append(model)
            return model

        frame = candles()
        frame["time"] -= pd.Timedelta(days=365)
        frame["close_time"] -= 365 * 86400000
        with patch.object(server, "load_file", return_value=frame), patch.object(server, "GaussianHMM", side_effect=factory):
            results, series = server.run_hmm("btc", "15m", "Q1", 3)
        self.assertEqual(len(models), 4)
        self.assertTrue(all(m.fit.call_count == 1 for m in models))
        return results, series

    def test_quarterly_metrics_reuse_fits_and_are_explicitly_in_sample(self):
        results, series = self.quarterly_result()
        self.assertFalse(any("error" in r for r in results))
        for result in results:
            profiles = result["regime_characteristics"]
            self.assertEqual([p["regime"] for p in profiles], STATE_ORDER)
            self.assertEqual(sum(p["n_candles"] for p in profiles), result["n_obs"])
            self.assertAlmostEqual(sum(p["proportion"] for p in profiles), 1)
            self.assertEqual(
                [p["n_candles"] for p in profiles],
                [series["models"][result["model"][-1]].count(tier) for tier in range(3)],
            )
        self.assertNotIn("forecast_comparison", series)
        comparison = series["distribution_comparison"]
        self.assertEqual(comparison["method"], "in_sample_quarter_fit")
        self.assertFalse(comparison["is_out_of_sample"])
        self.assertFalse(comparison["past_only_forecasts_available"])
        self.assertEqual(comparison["timestamps"], series["timestamps"])
        for mid in "BCD":
            columns = comparison["pairs"][mid]
            rows = [dict(zip(columns, values)) for values in zip(*columns.values())]
            self.assertEqual(summarize_comparisons(rows)["n_compared"], 70)
            self.assertTrue(all(0 <= r["jsd_bits"] <= 1 for r in rows))
        json.dumps(series, allow_nan=False)

    def test_degenerate_quarterly_fit_remains_a_gap_in_probability_metrics(self):
        results, series = self.quarterly_result(degenerate_model="B")
        self.assertTrue(results[1]["degenerate"])
        self.assertTrue(all(v is None for v in series["models"]["B"]))
        self.assertTrue(all(v is None for v in series["distribution_comparison"]["pairs"]["B"]["jsd_bits"]))
        self.assertTrue(all(v is not None for v in series["distribution_comparison"]["pairs"]["C"]["jsd_bits"]))


if __name__ == "__main__":
    unittest.main()
