#!/usr/bin/env python3
"""Capture the real dashboard's Regime Comparison panel, without rewriting data.

First run: python scripts/capture_regime_comparison.py --allow-analysis
Rerun:     python scripts/capture_regime_comparison.py

The first command permits the application's existing /api/run analysis once per
uncached coin/interval. Subsequent browser requests replay those exact responses.
Without --allow-analysis, a missing or stale response is a reported failure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COIN_LABELS = {"BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "Binance Coin",
               "DOGE": "Dogecoin", "LTC": "Litecoin"}
DEFAULT_COINS = ["BTC", "ETH", "DOGE", "LTC"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
INTERVALS = ["5m", "15m", "30m"]
STATE_ORDER = ["Low volatility", "Medium volatility", "High volatility"]
MODELS = "ABCD"
LOG = logging.getLogger("regime_screenshots")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_fingerprint(coin, interval):
    """Track the input files and inference code; never modify them."""
    raw = ROOT / "src/HMM/data/raw"
    paths = [ROOT / "src/HMM/server.py", ROOT / "src/HMM/forecast_comparison.py"]
    names = {p.name.lower(): p for p in raw.iterdir() if p.is_file()}
    for quarter in QUARTERS:
        name = f"{coin.lower()}_{interval}_{quarter}_2025.json".lower()
        require(name in names, f"Missing source data: {name}")
        paths.append(names[name])
    hashes = {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in paths}
    return sha256(json.dumps(hashes, sort_keys=True).encode())


def validate_response(data, coin, interval, quarter=None):
    require(isinstance(data, dict), "Analysis response is not an object")
    require(data.get("coin") == coin, f"Wrong response coin: {data.get('coin')}, expected {coin}")
    require(data.get("tf") == interval, f"Wrong response interval: {data.get('tf')}, expected {interval}")
    require(data.get("n_states") == 3, "Expected the application's three-state analysis")
    if quarter is None:
        return
    require(quarter in data.get("available", []), f"{quarter} has no available analysis")
    series = data.get("regime_series", {}).get(quarter)
    require(isinstance(series, dict), f"Missing {quarter} regime series")
    times, prices = series.get("price_timestamps", []), series.get("close_prices", [])
    require(len(times) > 20 and len(times) == len(prices), "Missing or misaligned price candles")
    require(all(isinstance(t, (int, float)) and math.isfinite(t) for t in times), "Invalid price timestamp")
    require(all(a < b for a, b in zip(times, times[1:])), "Price timestamps are duplicated or unordered")
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in prices), "Invalid closing price")
    state_times = series.get("timestamps", [])
    require(len(state_times) > 20 and all(a < b for a, b in zip(state_times, state_times[1:])),
            "State timestamps are missing, duplicated, or unordered")
    require(times[0] <= state_times[0] <= state_times[-1] <= times[-1], "State times fall outside price history")
    for mid in MODELS:
        rows = [r for r in data.get("results", []) if r.get("quarter") == quarter and r.get("model") == f"Model {mid}"]
        require(len(rows) == 1, f"Missing or duplicated Model {mid} result for {quarter}")
        row = rows[0]
        require(not row.get("error") and not row.get("degenerate"), f"Model {mid} fit failed: {row.get('error')}")
        require(row.get("coin") == coin and row.get("timeframe") == interval, f"Wrong Model {mid} result identity")
        require(row.get("n_obs") == len(state_times), f"Model {mid} observation count does not match its states")
        states = series.get("models", {}).get(mid, [])
        require(len(states) == len(state_times) and all(type(s) is int and s in (0, 1, 2) for s in states),
                f"Model {mid} has missing or invalid regime assignments")
    comparison = series.get("distribution_comparison", {})
    require(comparison.get("method") == "in_sample_quarter_fit", "Missing actual quarterly probability comparisons")
    require(comparison.get("state_order") == STATE_ORDER, "Probability state order is incorrect")
    require(comparison.get("timestamps") == state_times, "Probability comparisons are not aligned to regime times")
    require(comparison.get("closed") == [True] * len(state_times), "Incomplete historical comparison origins")
    for mid in "BCD":
        columns = comparison.get("pairs", {}).get(mid, {})
        for metric in ("overlap_percent", "jsd_bits", "kl_bits", "top1_agreement"):
            values = columns.get(metric, [])
            require(len(values) == len(state_times), f"Missing A-{mid} {metric} values")
            for value in values:
                if metric == "kl_bits" and value == "Infinity":
                    continue
                require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                        f"Invalid A-{mid} {metric} value")
                upper = {"overlap_percent": 100, "jsd_bits": 1, "top1_agreement": 1}.get(metric)
                require(upper is None or value <= upper, f"A-{mid} {metric} exceeds its range")


class ResponseCache:
    def __init__(self, directory, base_url, allow_analysis=False, timeout=1800):
        self.directory = directory
        self.base_url = base_url.rstrip("/")
        self.allow_analysis = allow_analysis
        self.timeout = timeout
        self.ready = {}

    def get(self, coin, interval):
        fingerprint = source_fingerprint(coin, interval)
        path = self.directory / f"{coin}_{interval}.json.gz"
        metadata_path = path.with_suffix(".meta.json")
        url = self.base_url + "/api/run?" + urlencode({"coin": coin.lower(), "tf": interval, "states": 3})
        cached_error = "No saved application response"
        if path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                body = gzip.decompress(path.read_bytes())
                require(metadata["source_fingerprint"] == fingerprint, "Source data or analysis code changed")
                require(metadata["source_url"] == url, "Saved response belongs to another application URL")
                require(metadata["response_sha256"] == sha256(body), "Saved response checksum mismatch")
                data = json.loads(body)
                validate_response(data, coin, interval)
                self.ready[(coin.lower(), interval)] = path
                LOG.info("Reusing saved response: %s %s", coin, interval)
                return data, metadata
            except (ValueError, KeyError, OSError) as exc:
                cached_error = str(exc)
        require(self.allow_analysis,
                f"{cached_error} for {coin} {interval}. Run with --allow-analysis to use the normal application analysis once.")
        LOG.info("Loading real /api/run result: %s %s (may take several minutes)", coin, interval)
        started = time.monotonic()
        with urlopen(url, timeout=self.timeout) as response:
            body = response.read()
        data = json.loads(body)
        validate_response(data, coin, interval)
        require(source_fingerprint(coin, interval) == fingerprint, "Source data changed during analysis")
        metadata = {"source_url": url, "captured_at": utc_now(), "source_fingerprint": fingerprint,
                    "response_sha256": sha256(body), "analysis_seconds": round(time.monotonic() - started, 2)}
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(gzip.compress(body))
        temporary.replace(path)
        write_json(metadata_path, metadata)
        self.ready[(coin.lower(), interval)] = path
        LOG.info("Cached unchanged API response: %s %s (%.1fs)", coin, interval, metadata["analysis_seconds"])
        return data, metadata

    def route(self, route):
        """Replay exact real responses; browser interactions never trigger fits."""
        parsed = urlsplit(route.request.url)
        headers = {"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"}
        try:
            require(route.request.method == "GET", "Only read-only GET requests are allowed")
            if parsed.path == "/api/run":
                args = parse_qs(parsed.query)
                key = (args.get("coin", [""])[0].lower(), args.get("tf", [""])[0])
                require(args.get("states", ["3"])[0] == "3", "Only three-state responses were cached")
                require(key in self.ready, f"No prepared response for {key[0]} {key[1]}")
                body = gzip.decompress(self.ready[key].read_bytes())
            else:
                require(parsed.path in ("/api/coins", "/api/files"), "Live or training endpoints are disabled during capture")
                with urlopen(self.base_url + parsed.path + ("?" + parsed.query if parsed.query else ""), timeout=30) as response:
                    body = response.read()
            route.fulfill(status=200, body=body, content_type="application/json", headers=headers)
        except Exception as exc:
            route.fulfill(status=409, body=json.dumps({"error": str(exc)}), content_type="application/json", headers=headers)


def validate_png(body, bounds, scale):
    from PIL import Image, ImageStat

    with Image.open(io.BytesIO(body)) as image:
        image.verify()
    with Image.open(io.BytesIO(body)) as image:
        require(image.format == "PNG", "Capture is not PNG")
        width, height = image.size
        require(abs(width - bounds["width"] * scale) <= 3 and abs(height - bounds["height"] * scale) <= 3,
                "Screenshot does not match the complete section bounds")
        require(width >= 1500 and height >= 1500, "Screenshot resolution is too small for the complete panel")
        require(max(ImageStat.Stat(image.convert("RGB")).stddev) > 8, "Screenshot appears empty")
        return {"width": width, "height": height}


def select_configuration(page, coin, interval, timeout):
    page.wait_for_function("document.querySelectorAll('.coin-tab').length >= 4", timeout=timeout)
    page.wait_for_function("!document.getElementById('run-btn').disabled", timeout=timeout)
    current = page.evaluate("({coin:activeCoin,tf:document.getElementById('tf-sel').value,loaded:!!lastData})")
    button = page.get_by_role("button", name=COIN_LABELS[coin], exact=True)
    require(button.count() == 1 and "unavail" not in (button.get_attribute("class") or ""),
            f"{coin} has no selectable dashboard control")
    if current["coin"] != coin.lower():
        # Set both form values before the existing coin handler submits one request.
        page.locator("#tf-sel").evaluate("(select,value)=>{select.value=value}", interval)
        button.click()
    elif current["tf"] != interval:
        page.locator("#tf-sel").select_option(interval)
    elif not current["loaded"]:
        page.locator("#run-btn").click()
    page.wait_for_function("""({coin,interval}) =>
      !document.getElementById('run-btn').disabled && lastData?.coin === coin &&
      lastData.tf === interval && activeCoin === coin.toLowerCase() &&
      document.getElementById('tf-sel').value === interval &&
      comparisonView.key?.toLowerCase() === coin.toLowerCase()+':'+interval+':history'""",
      arg={"coin": coin, "interval": interval}, timeout=timeout)
    if not page.locator("#panel-regimes").is_visible():
        page.locator("#tab-regimes").click()


def capture_quarter(page, coin, interval, quarter, timeout):
    page.locator("#view-window").select_option(quarter)
    page.wait_for_function("""({coin,interval,quarter}) => {
      const range=comparisonView.quarterWindows?.[quarter];
      if(!range || comparisonView.frame !== null || lastData?.coin !== coin || lastData.tf !== interval ||
         document.getElementById('view-window').value !== quarter ||
         comparisonView.start !== range.start || comparisonView.start+comparisonView.span !== range.end) return false;
      return ['a','b','c','d'].every(mid => {
        const chart=Chart.getChart(document.getElementById('comparison-'+mid));
        return chart && chart.chartArea.width>500 && chart.chartArea.height>100 &&
          chart.scales.x.min===range.start && chart.scales.x.max===range.end &&
          chart.data.datasets[0].data.some(point=>point.x>=range.start && point.x<=range.end && Number.isInteger(point.regime));
      });
    }""", arg={"coin": coin, "interval": interval, "quarter": quarter}, timeout=timeout)
    page.evaluate("() => document.fonts.ready")
    page.evaluate("() => new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))")
    page.mouse.move(0, 0)
    panel = page.locator("#panel-regimes")
    require(panel.locator("canvas").count() == 4, "Expected all four model charts inside Regime Comparison")
    require(panel.locator("#comparison-agreement table tbody tr").count() == 3, "Model comparison table is incomplete")
    cells = panel.locator("#comparison-agreement table tbody td").all_text_contents()
    require(cells and all(value.strip() not in ("", "—", "N/A") for value in cells), "Comparison table has unloaded values")
    require(not panel.evaluate("el=>[...el.querySelectorAll('.agreement-scroll')].some(node=>node.scrollWidth>node.clientWidth+1)"),
            "Model comparison table is horizontally clipped")
    bounds = panel.bounding_box()
    require(bounds and bounds["height"] > 900 and bounds["width"] > 900, "Regime Comparison section is incomplete or hidden")
    canvas_hashes = [sha256(canvas.evaluate("el=>el.toDataURL('image/png')").encode()) for canvas in panel.locator("canvas").all()]
    body = panel.screenshot(type="png", animations="disabled", caret="hide", scale="device", timeout=timeout)
    dimensions = validate_png(body, bounds, page.evaluate("devicePixelRatio"))
    return body, {**dimensions, "chart_sha256": canvas_hashes,
                  "visible_range": page.locator("#timeline-range").inner_text(),
                  "section_bounds_css": bounds}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:5050")
    parser.add_argument("--coins", nargs="+", type=str.upper, choices=list(COIN_LABELS), default=DEFAULT_COINS)
    parser.add_argument("--intervals", nargs="+", choices=INTERVALS, default=INTERVALS)
    parser.add_argument("--quarters", nargs="+", choices=QUARTERS, default=QUARTERS)
    parser.add_argument("--output", type=Path, default=ROOT / "screenshots/regime_comparison")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "screenshots/.regime_response_cache")
    parser.add_argument("--allow-analysis", action="store_true", help="Permit /api/run only for missing or stale saved responses")
    parser.add_argument("--analysis-timeout", type=int, default=1800, help="Seconds allowed for each normal API analysis")
    parser.add_argument("--ui-timeout", type=int, default=60, help="Seconds allowed for browser rendering")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args(argv)
    if urlsplit(args.base_url).scheme not in ("http", "https"):
        parser.error("--base-url must be an HTTP(S) application URL")
    for key in ("coins", "intervals", "quarters"):
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    require(args.analysis_timeout > 0 and args.ui_timeout > 0, "Timeouts must be positive")
    return args


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.output / "capture.log")], force=True)
    configurations = [(coin, quarter, interval) for coin in args.coins for quarter in args.quarters for interval in args.intervals]
    report = {"started_at": utc_now(), "base_url": args.base_url, "expected": len(configurations),
              "section_selector": "#panel-regimes", "device_scale_factor": 2, "results": []}
    report_path = args.output / "capture_report.json"
    completed, image_hashes, data_hashes, canvas_hashes = set(), {}, {}, {}

    def record(coin, quarter, interval, error=None, **details):
        key = (coin, quarter, interval)
        if key in completed:
            return
        completed.add(key)
        result = {"coin": coin, "quarter": quarter, "interval": interval,
                  "status": "failed" if error else "success", **details}
        if error:
            result["error"] = str(error)
        report["results"].append(result)
        report["successful"] = sum(r["status"] == "success" for r in report["results"])
        report["failed"] = sum(r["status"] == "failed" for r in report["results"])
        write_json(report_path, report)
        LOG.log(logging.ERROR if error else logging.INFO, "%s %s %s: %s", coin, quarter, interval, error or "saved")

    browser = None
    try:
        from playwright.sync_api import sync_playwright

        with urlopen(args.base_url.rstrip("/") + "/api/coins", timeout=30) as response:
            available = json.load(response)
        require(isinstance(available, list), "Application coin list is invalid")
        cache = ResponseCache(args.cache_dir, args.base_url, args.allow_analysis, args.analysis_timeout)
        pairs = [(c, tf) for c in args.coins for tf in args.intervals]
        # The application's first automatic request is BTC 15m; use it when requested.
        if ("BTC", "15m") in pairs:
            pairs.remove(("BTC", "15m"))
            pairs.insert(0, ("BTC", "15m"))
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=not args.headed)
            context = browser.new_context(viewport={"width": 1800, "height": 1000}, device_scale_factor=2,
                                          locale="en-US", timezone_id="UTC", color_scheme="light",
                                          reduced_motion="reduce", service_workers="block")
            context.route("**/api/**", cache.route)
            page = None
            for coin, interval in pairs:
                try:
                    require(coin.lower() in available, f"{coin} is unsupported or has no data; available coins: {', '.join(available)}")
                    data, metadata = cache.get(coin, interval)
                    if page is None or page.is_closed():
                        page = context.new_page()
                        page.on("pageerror", lambda error: LOG.warning("Browser: %s", error))
                        page.goto(args.base_url, wait_until="domcontentloaded", timeout=args.ui_timeout * 1000)
                    select_configuration(page, coin, interval, args.ui_timeout * 1000)
                except Exception as exc:
                    if page is not None and not page.is_closed():
                        try:
                            LOG.error("Browser state: %s", page.evaluate("""() => ({
                              status:document.getElementById('status')?.textContent,
                              busy:document.getElementById('run-btn')?.disabled,
                              coin:typeof activeCoin==='undefined' ? null : activeCoin,
                              loaded:typeof lastData==='undefined' ? null : {coin:lastData?.coin,tf:lastData?.tf},
                              view:typeof comparisonView==='undefined' ? null : comparisonView.key
                            })"""))
                        except Exception:
                            pass
                    for quarter in args.quarters:
                        record(coin, quarter, interval, error=exc)
                    if page is not None:
                        page.close()
                        page = None
                    continue
                for quarter in args.quarters:
                    try:
                        validate_response(data, coin, interval, quarter)
                        series = data["regime_series"][quarter]
                        data_hash = sha256(json.dumps({k: series[k] for k in ("price_timestamps", "close_prices", "timestamps", "models")},
                                                     sort_keys=True, separators=(",", ":")).encode())
                        require(data_hash not in data_hashes, f"Duplicate chart data from {data_hashes.get(data_hash)}")
                        for attempt in range(2):
                            try:
                                body, details = capture_quarter(page, coin, interval, quarter, args.ui_timeout * 1000)
                                break
                            except Exception:
                                if attempt:
                                    raise
                                LOG.warning("Retrying browser rendering only: %s %s %s", coin, quarter, interval)
                                page.evaluate("updateComparisonCharts()")
                        image_hash = sha256(body)
                        require(image_hash not in image_hashes, f"Duplicate screenshot from {image_hashes.get(image_hash)}")
                        charts_key = tuple(details["chart_sha256"])
                        require(charts_key not in canvas_hashes, f"Charts did not change from {canvas_hashes.get(charts_key)}")
                        require(source_fingerprint(coin, interval) == metadata["source_fingerprint"], "Source inputs changed during capture")
                        target = args.output / coin / quarter / f"{coin}_{quarter}_{interval}.png"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix(".png.tmp")
                        temporary.write_bytes(body)
                        temporary.replace(target)
                        image_hashes[image_hash] = data_hashes[data_hash] = canvas_hashes[charts_key] = f"{coin}/{quarter}/{interval}"
                        record(coin, quarter, interval, path=str(target), sha256=image_hash, data_sha256=data_hash,
                               api_response=metadata, **details)
                    except Exception as exc:
                        record(coin, quarter, interval, error=exc)
            browser.close()
            browser = None
    except (Exception, KeyboardInterrupt) as exc:
        LOG.error("Capture stopped: %s", exc)
        for coin, quarter, interval in configurations:
            record(coin, quarter, interval, error=str(exc) or "Interrupted")
    report["finished_at"] = utc_now()
    write_json(report_path, report)
    LOG.info("TOTAL: %s successful, %s failed, %s expected. Report: %s",
             report.get("successful", 0), report.get("failed", 0), report["expected"], report_path)
    return 1 if report.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(run(arguments()))
