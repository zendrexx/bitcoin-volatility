import requests
import json
import time
from datetime import datetime

BASE_URL = "https://api.binance.com/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVALS = ["5m", "15m", "30m"]
LIMIT = 1000  # max candles per request

YEAR = 2025

# Quarterly date ranges
QUARTERS = {
    "Q1": (
        datetime(YEAR, 1, 1),
        datetime(YEAR, 3, 31, 23, 59, 59)
    ),
    "Q2": (
        datetime(YEAR, 4, 1),
        datetime(YEAR, 6, 30, 23, 59, 59)
    ),
    "Q3": (
        datetime(YEAR, 7, 1),
        datetime(YEAR, 9, 30, 23, 59, 59)
    ),
    "Q4": (
        datetime(YEAR, 10, 1),
        datetime(YEAR, 12, 31, 23, 59, 59)
    ),
}


def get_klines(interval, start_time, end_time):
    all_data = []

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    while start_ts < end_ts:

        params = {
            "symbol": SYMBOL,
            "interval": interval,
            "limit": LIMIT,
            "startTime": start_ts,
            "endTime": end_ts
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print("Error:", response.text)
            break

        data = response.json()

        if not data:
            break

        all_data.extend(data)

        # move to next candle
        start_ts = data[-1][0] + 1

        print(f"{interval} -> {len(all_data)} candles fetched")

        # avoid rate limit
        time.sleep(0.2)

    return all_data


def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    for quarter_name, (start_time, end_time) in QUARTERS.items():

        print(f"\n========== {quarter_name} ==========")

        for interval in INTERVALS:

            print(f"\nFetching {interval} data...")

            klines = get_klines(interval, start_time, end_time)

            filename = f"btc_{interval}_{quarter_name}_{YEAR}.json"

            save_json(klines, filename)

            print(f"Saved {len(klines)} rows -> {filename}")