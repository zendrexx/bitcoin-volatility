from data_loader import load_binance_json
from feature_engineering import add_features
from hmm_model import train_hmm, predict_states
from export_to_json import export_json
# load
df = load_binance_json("data/raw/btc_5m_1year.json")

df, X = add_features(df)

print("Total rows:", len(df))

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
    
print(df.head(50))
#export_json(df, "5m")