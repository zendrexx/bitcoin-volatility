"""
feature_engineering.py
Computes features for HMM training.
Provides four feature sets (Models A–D) for the rolling window comparison.

Model A: Log Return + High-Low Range
Model B: Log Return + High-Low Range + Volume
Model C: Log Return + High-Low Range + Rolling Volatility
Model D: Log Return + High-Low Range + Volume + Rolling Volatility

Fixed:
- Duplicate function removed (was defined twice in original)
- Each model gets its own function for clarity
- StandardScaler applied per-model
- Rolling volatility window made configurable
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _base_features(df: pd.DataFrame, vol_window: int = 30) -> pd.DataFrame:
    """Compute shared derived columns in-place and return cleaned df."""
    df = df.copy()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["rolling_vol"]    = df["log_return"].rolling(vol_window).std()
    df["log_volume"]     = np.log1p(df["volume"])  
    df = df.dropna().reset_index(drop=True)
    return df


def build_features_A(df: pd.DataFrame, vol_window: int = 10):
    """Model A: Log Return + High-Low Range"""
    df = _base_features(df, vol_window)
    X = np.column_stack([df["log_return"], df["high_low_range"]])
    X = StandardScaler().fit_transform(X)
    return df, X


def build_features_B(df: pd.DataFrame, vol_window: int = 10):
    """Model B: Log Return + High-Low Range + Log Volume"""
    df = _base_features(df, vol_window)
    X = np.column_stack([df["log_return"], df["high_low_range"], df["log_volume"]])  
    X = StandardScaler().fit_transform(X)
    return df, X


def build_features_C(df: pd.DataFrame, vol_window: int = 10):
    """Model C: Log Return + High-Low Range + Rolling Volatility"""
    df = _base_features(df, vol_window)
    X = np.column_stack([df["log_return"], df["high_low_range"], df["rolling_vol"]])
    X = StandardScaler().fit_transform(X)
    return df, X


def build_features_D(df: pd.DataFrame, vol_window: int = 10):
    """Model D: Log Return + High-Low Range + Log Volume + Rolling Volatility"""
    df = _base_features(df, vol_window)
    X = np.column_stack([
        df["log_return"],
        df["high_low_range"],
        df["log_volume"],     
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