from sklearn.covariance import log_likelihood

from data_loader import load_binance_json
from feature_engineering import add_features
from hmm_model import train_hmm, predict_states
from export_to_json import export_json


timeframes = ["5m", "15m", "30m"]
quarters = ["Q1", "Q2", "Q3", "Q4"]
results = []


for tf in timeframes:
    for q in quarters:
    
        file_path = f"data/raw/btc_{tf}_{q}_2025.json"
        # load
        df = load_binance_json(file_path)

        df, X = add_features(df)

        print("Total rows:", len(df))
        print("time frame:", tf)
        print("quarter:", q)
        #split = int(len(X) * 0.8)

        #X_train = X[:split]
        #X_test = X[split:]

        #df_train = df.iloc[:split]
        #df_test = df.iloc[split:]

        # train
        model = train_hmm(X)

        # predict
        df["state"] = predict_states(model, X)
        #df_test["state"] = predict_states(model, X_test)
        df = df[['time', 'open', 'high', 'low', 'close', 'volume', 'log_return', 'volatility', 'state']]
        vol_means = df.groupby("state")["volatility"].mean()

        sorted_states = vol_means.sort_values().index

        state_map = {
            sorted_states[0]: "Low",
            sorted_states[1]: "Medium",
            sorted_states[2]: "High"
        }
        log_likelihood = model.score(X)
         # save result
        results.append({
            "timeframe": tf,
            "quarter": q,
            "log_likelihood": log_likelihood,
            "rows": len(df)
        })

        print("Log Likelihood:", log_likelihood)

        df["regime"] = df["state"].map(state_map)
        #print(df.head(50))
        print(df.groupby("state")["volatility"].describe())
        print(df.groupby("state")["log_return"].describe())
        #export_json(df, "5m")

print("\nFINAL RESULTS\n")

for r in results:
    print(
        f"{r['timeframe']} {r['quarter']} | "
        f"LL: {r['log_likelihood']:.2f} | "
        f"Rows: {r['rows']}"
    )