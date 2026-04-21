import requests
import json
import time
from datetime import datetime, timedelta

BASE_URL = "https://api.binance.com/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVALS = ["5m", "15m", "30m"]
LIMIT = 1000  # max per request

# 1 year range
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=365)


def get_klines(interval):
    all_data = []

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol": SYMBOL,
            "interval": interval,
            "limit": LIMIT,
            "startTime": start_ts
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print("Error:", response.text)
            break

        data = response.json()

        if not data:
            break

        all_data.extend(data)

        # move forward using last candle open time
        start_ts = data[-1][0] + 1

        print(f"{interval} → {len(all_data)} candles")

        # avoid rate limit
        time.sleep(0.2)

    return all_data


def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    for interval in INTERVALS:
        print(f"\nFetching {interval}...")

        klines = get_klines(interval)

        filename = f"btc_{interval}_1year.json"
        save_json(klines, filename)

        print(f"Saved {len(klines)} rows → {filename}")