"""
data_fetcher.py
Fetches OHLCV klines from Binance for any symbol.
Saves as: data/raw/{coin}_{interval}_{quarter}_{year}.json
e.g.  btc_5m_Q1_2025.json
      eth_15m_Q2_2025.json
      doge_30m_Q3_2025.json

Usage:
    python data_fetcher.py            # fetches BTC (default)
    python data_fetcher.py eth        # fetches ETH
    python data_fetcher.py doge       # fetches DOGE
    python data_fetcher.py ltc        # fetches LTC
    python data_fetcher.py btc eth doge ltc  # fetches all
"""

import requests
import json
import time
import os
import sys
from datetime import datetime

BASE_URL   = "https://api.binance.com/api/v3/klines"
INTERVALS  = ["5m", "15m", "30m"]
LIMIT      = 1000
YEAR       = 2025
OUTPUT_DIR = "data/raw"

# Map short coin name → Binance symbol
COIN_MAP = {
    "btc":  "BTCUSDT",
    "eth":  "ETHUSDT",
    "doge": "DOGEUSDT",
    "ltc":  "LTCUSDT",
}

QUARTERS = {
    "Q1": (datetime(YEAR, 1, 1),  datetime(YEAR, 3, 31, 23, 59, 59)),
    "Q2": (datetime(YEAR, 4, 1),  datetime(YEAR, 6, 30, 23, 59, 59)),
    "Q3": (datetime(YEAR, 7, 1),  datetime(YEAR, 9, 30, 23, 59, 59)),
    "Q4": (datetime(YEAR, 10, 1), datetime(YEAR, 12, 31, 23, 59, 59)),
}


def get_klines(symbol: str, interval: str, start_time: datetime,
               end_time: datetime, retries: int = 3) -> list:
    all_data = []
    start_ts = int(start_time.timestamp() * 1000)
    end_ts   = int(end_time.timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "limit":     LIMIT,
            "startTime": start_ts,
            "endTime":   end_ts,
        }

        for attempt in range(1, retries + 1):
            try:
                response = requests.get(BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"  [Attempt {attempt}/{retries}] Error: {e}")
                if attempt == retries:
                    print("  Max retries reached. Stopping.")
                    return all_data
                time.sleep(2 ** attempt)

        data = response.json()
        if not data:
            break

        all_data.extend(data)
        start_ts = data[-1][0] + 1
        print(f"  {interval} -> {len(all_data)} candles fetched")
        time.sleep(0.2)

    return all_data


def save_json(data: list, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def fetch_coin(coin: str):
    coin = coin.lower()
    if coin not in COIN_MAP:
        print(f"Unknown coin '{coin}'. Choose from: {', '.join(COIN_MAP.keys())}")
        return

    symbol = COIN_MAP[coin]
    print(f"\n{'='*50}")
    print(f"Fetching {coin.upper()} ({symbol})")
    print(f"{'='*50}")

    for quarter_name, (start_time, end_time) in QUARTERS.items():
        print(f"\n  -- {quarter_name} --")
        for interval in INTERVALS:
            print(f"  Fetching {interval}...")
            klines   = get_klines(symbol, interval, start_time, end_time)
            filename = os.path.join(OUTPUT_DIR, f"{coin}_{interval}_{quarter_name}_{YEAR}.json")
            save_json(klines, filename)
            print(f"  Saved {len(klines)} rows → {filename}")


if __name__ == "__main__":
    # Default to BTC if no args given
    coins = sys.argv[1:] if len(sys.argv) > 1 else ["btc"]
    for c in coins:
        fetch_coin(c)