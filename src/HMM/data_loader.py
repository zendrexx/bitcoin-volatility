import json
import pandas as pd


KLINE_COLUMNS = [
    "time",       # open time (ms)
    "open",
    "high",
    "low",
    "close",
    "volume",     # total BTC traded during interval
    "close_time",
    "qav",        # Quote Asset Volume (total value in USDT)
    "trades",     # number of trades
    "tbbav",      # Taker Buy Base Asset Volume
    "tbqav",      # Taker Buy Quote Asset Volume
    "ignore",
]

NUMERIC_COLS = ["open", "high", "low", "close", "volume", "qav", "tbbav", "tbqav"]


def load_binance_json(filepath: str) -> pd.DataFrame:
  
    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data, columns=KLINE_COLUMNS)

    # Parse timestamps
    df["time"]       = pd.to_datetime(df["time"],       unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # Cast all numeric columns (Binance returns them as strings)
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(float)

    df["trades"] = df["trades"].astype(int)

    return df.sort_values("time").reset_index(drop=True)