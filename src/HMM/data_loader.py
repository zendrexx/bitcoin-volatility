import json
import pandas as pd

def load_binance_json(filepath):#pass 5,15,30
    with open(filepath, "r") as f:#read json file
        data = json.load(f)

    df = pd.DataFrame(data, columns=[#base sa binibigay ni binance to doon sa json
        "time",#open time
        "open",
        "high",
        "low",
        "close",
        "volume",#total btc traded during interval
        "close_time",
        "qav",#Quote Asset Volume, Total traded value in quote currency
        "trades",#Number of Trades, Total executed trades during interval
        "tbbav",#Taker Buy Base Asset Volume, Amount bought by aggressive buyers
        "tbqav",#Taker Buy Quote Asset Volume,Value of aggressive buy orders in USDT
        "ignore"
        
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["close"] = df["close"].astype(float)#baka string kasi 

    return df.sort_values("time")#return it na sorted na