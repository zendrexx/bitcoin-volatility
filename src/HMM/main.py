from data_loader import load_binance_json
from feature_engineering import add_features
from hmm_model import train_hmm, predict_states
from export_to_json import export_json
# load
df = load_binance_json("data/raw/btc_5m_1year.json")

# features (THIS already creates X)
df, X = add_features(df)

print("Total rows:", len(df))

# train
model = train_hmm(X)

# predict
df["state"] = predict_states(model, X)

print(df.head())
export_json(df, "5m")