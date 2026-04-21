import numpy as np
from sklearn.preprocessing import StandardScaler

def add_features(df):
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))#compute log return log(Pt/Pt-1)


    df["volatility"] = df["log_return"].rolling(10).std()#absolute return

    df = df.dropna()#remove Nan, walang prev row ang first row

    X = np.column_stack([#return1, volatility1
        df["log_return"],
        df["volatility"]
    ])

    X = StandardScaler().fit_transform(X)

    return df, X