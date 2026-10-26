import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import capture_regime_comparison as capture


def response_fixture():
    times = list(range(30))
    columns = {"overlap_percent": [100] * 30, "jsd_bits": [0] * 30,
               "kl_bits": ["Infinity"] * 30, "top1_agreement": [1] * 30}
    return {"coin": "BTC", "tf": "15m", "n_states": 3, "available": ["Q1"],
            "results": [{"coin": "BTC", "timeframe": "15m", "quarter": "Q1",
                         "model": f"Model {mid}", "n_obs": 30} for mid in "ABCD"],
            "regime_series": {"Q1": {"price_timestamps": times, "close_prices": [100] * 30,
                "timestamps": times, "models": {mid: [0, 1, 2] * 10 for mid in "ABCD"},
                "distribution_comparison": {"method": "in_sample_quarter_fit",
                    "state_order": capture.STATE_ORDER, "timestamps": times,
                    "closed": [True] * 30, "pairs": {mid: copy.deepcopy(columns) for mid in "BCD"}}}}}


class CaptureValidationTests(unittest.TestCase):
    def test_defaults_have_48_configurations_and_the_confirmed_eth_coin(self):
        args = capture.arguments([])
        self.assertEqual(len(args.coins) * len(args.quarters) * len(args.intervals), 48)
        self.assertIn("ETH", args.coins)
        self.assertNotIn("BNB", args.coins)
        self.assertFalse(args.allow_analysis)

    def test_response_identity_and_complete_comparisons_are_required(self):
        data = response_fixture()
        capture.validate_response(data, "BTC", "15m", "Q1")
        with self.assertRaisesRegex(ValueError, "Wrong response coin"):
            capture.validate_response(data, "ETH", "15m", "Q1")
        data["regime_series"]["Q1"]["models"]["A"][2] = None
        with self.assertRaisesRegex(ValueError, "invalid regime"):
            capture.validate_response(data, "BTC", "15m", "Q1")

    def test_failed_fit_and_missing_probability_columns_prevent_success(self):
        data = response_fixture()
        data["results"][0]["error"] = "collapsed state"
        with self.assertRaisesRegex(ValueError, "fit failed"):
            capture.validate_response(data, "BTC", "15m", "Q1")
        del data["results"][0]["error"]
        data["regime_series"]["Q1"]["distribution_comparison"]["pairs"]["B"]["jsd_bits"] = []
        with self.assertRaisesRegex(ValueError, "Missing A-B jsd_bits"):
            capture.validate_response(data, "BTC", "15m", "Q1")

    def test_cache_requires_opt_in_then_replays_original_bytes_without_network(self):
        body = json.dumps(response_fixture()).encode()
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(capture, "source_fingerprint", return_value="inputs-v1"), \
             patch.object(capture, "urlopen", return_value=io.BytesIO(body)) as network:
            path = Path(temporary)
            cache = capture.ResponseCache(path, "http://localhost:5050")
            with self.assertRaisesRegex(ValueError, "--allow-analysis"):
                cache.get("BTC", "15m")
            network.assert_not_called()
            data, metadata = capture.ResponseCache(path, "http://localhost:5050", True).get("BTC", "15m")
            self.assertEqual(network.call_count, 1)
            self.assertEqual(gzip.decompress((path / "BTC_15m.json.gz").read_bytes()), body)
            cached, saved_metadata = cache.get("BTC", "15m")
            self.assertEqual(data, cached)
            self.assertEqual(metadata, saved_metadata)
            self.assertEqual(network.call_count, 1)
            with patch.object(capture, "source_fingerprint", return_value="inputs-v2"):
                with self.assertRaisesRegex(ValueError, "Source data or analysis code changed"):
                    cache.get("BTC", "15m")
            self.assertEqual(network.call_count, 1)

    def test_unprepared_browser_requests_never_trigger_fitting(self):
        route = Mock()
        route.request.url = "http://localhost:5050/api/run?coin=eth&tf=5m&states=3"
        route.request.method = "GET"
        with tempfile.TemporaryDirectory() as temporary, patch.object(capture, "urlopen") as network:
            capture.ResponseCache(Path(temporary), "http://localhost:5050", True).route(route)
            network.assert_not_called()
            self.assertEqual(route.fulfill.call_args.kwargs["status"], 409)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Install requirements-screenshots.txt for PNG validation")
    def test_blank_or_incomplete_pngs_are_rejected(self):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (2000, 2000), "white")
        body = io.BytesIO()
        image.save(body, "PNG")
        bounds = {"width": 1000, "height": 1000}
        with self.assertRaisesRegex(ValueError, "appears empty"):
            capture.validate_png(body.getvalue(), bounds, 2)
        ImageDraw.Draw(image).rectangle((200, 200, 1800, 1800), fill="green")
        body = io.BytesIO()
        image.save(body, "PNG")
        self.assertEqual(capture.validate_png(body.getvalue(), bounds, 2), {"width": 2000, "height": 2000})
        with self.assertRaisesRegex(ValueError, "complete section bounds"):
            capture.validate_png(body.getvalue(), {"width": 1100, "height": 1000}, 2)


if __name__ == "__main__":
    unittest.main()
