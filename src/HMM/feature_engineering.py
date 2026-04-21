import numpy as np
from sklearn.preprocessing import StandardScaler

def add_features(df):
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # better volatility proxy (less smoothing)
    df["volatility"] = df["log_return"].abs()

    df = df.dropna()

    X = np.column_stack([
        df["log_return"],
        df["volatility"]
    ])

    X = StandardScaler().fit_transform(X)

    return df, X