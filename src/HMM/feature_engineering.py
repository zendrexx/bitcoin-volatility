import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



EXCLUDED_WINDOWS = []


def _drop_excluded(df: pd.DataFrame) -> pd.DataFrame:

    if "time" not in df.columns:
        return df
    keep = pd.Series(True, index=df.index)
    for start, end in EXCLUDED_WINDOWS:
        keep &= ~df["time"].between(pd.Timestamp(start), pd.Timestamp(end))
    return df[keep].reset_index(drop=True)


def _base_features(df: pd.DataFrame, vol_window: int = 30) -> pd.DataFrame:
    """Compute shared derived columns in-place and return cleaned df."""
    df = df.copy()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["rolling_vol"]    = df["log_return"].rolling(vol_window).std()
    df = df.dropna().reset_index(drop=True)
    return _drop_excluded(df)


def build_features_A(df: pd.DataFrame, vol_window: int = 10):
    """Model A: Log Return + High-Low Range"""
    df = _base_features(df, vol_window)
    X = np.column_stack([df["log_return"], df["high_low_range"]])
    X = StandardScaler().fit_transform(X)
    return df, X


def build_features_B(df: pd.DataFrame, vol_window: int = 10):
    """Model B: Log Return + High-Low Range + Volume"""
    df = _base_features(df, vol_window)
    X = np.column_stack([df["log_return"], df["high_low_range"], df["volume"]])
    X = StandardScaler().fit_transform(X)
    return df, X


def build_features_C(df: pd.DataFrame, vol_window: int = 10):
    """Model C: Log Return + High-Low Range + Rolling Volatility"""
    df = _base_features(df, vol_window)
    X = np.column_stack([df["log_return"], df["high_low_range"], df["rolling_vol"]])
    X = StandardScaler().fit_transform(X)
    return df, X


def build_features_D(df: pd.DataFrame, vol_window: int = 10):
    """Model D: Log Return + High-Low Range + Volume + Rolling Volatility"""
    df = _base_features(df, vol_window)
    X = np.column_stack([
        df["log_return"],
        df["high_low_range"],
        df["volume"],
        df["rolling_vol"],
    ])
    X = StandardScaler().fit_transform(X)
    return df, X


# Convenience map used by the rolling window runner
MODEL_BUILDERS = {
    "A": build_features_A,
    "B": build_features_B,
    "C": build_features_C,
    "D": build_features_D,
}