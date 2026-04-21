import json
import pandas as pd

def load_binance_json(filepath):#pass 5,15,30
    with open(filepath, "r") as f:#read json file
        data = json.load(f)

    df = pd.DataFrame(data, columns=[#base sa binibigay ni binance to doon sa json
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qav",
        "trades",
        "tbbav",
        "tbqav",
        "ignore"
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["close"] = df["close"].astype(float)#baka string kasi 

    return df.sort_values("time")#return it na sorted na