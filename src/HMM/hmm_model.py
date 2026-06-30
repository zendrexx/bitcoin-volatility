"""
hmm_model.py
Thin wrapper around hmmlearn's GaussianHMM.
Handles training, state prediction, and metric computation (LL, AIC, BIC).
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM


def train_hmm(X: np.ndarray, n_states: int = 3, n_iter: int = 500, random_state: int = 42) -> GaussianHMM:
    """Fit a Gaussian HMM on feature matrix X."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)
    return model


def predict_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Return the most likely hidden state sequence via Viterbi."""
    return model.predict(X)


def compute_metrics(model: GaussianHMM, X: np.ndarray) -> dict:
    """
    Compute log-likelihood, AIC, and BIC for a fitted model.

    AIC = 2k - 2*LL
    BIC = k*ln(n) - 2*LL
    where k = number of free parameters, n = number of observations.
    """
    n, n_features = X.shape
    n_states = model.n_components

    # Free parameters:
    #   transition matrix: n_states * (n_states - 1)   (rows sum to 1)
    #   start probs:       n_states - 1
    #   means:             n_states * n_features
    #   full covariances:  n_states * n_features * (n_features + 1) / 2
    k_trans   = n_states * (n_states - 1)
    k_start   = n_states - 1
    k_means   = n_states * n_features
    k_cov     = n_states * n_features * (n_features + 1) // 2
    k = k_trans + k_start + k_means + k_cov

    log_likelihood = model.score(X)
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood

    return {
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "n_params": k,
        "n_obs": n,
    }


def label_regimes(df, states: np.ndarray) -> dict:
    """
    Map raw integer states to interpretable regime labels
    by sorting on mean rolling volatility.
    Returns a dict {state_int: label_str} and adds a 'regime' column to df.
    """
    import pandas as pd

    df = df.copy()
    df["state"] = states
    vol_means = df.groupby("state")["rolling_vol"].mean().sort_values()

    n = len(vol_means)
    if n == 2:
        names = ["Low Vol", "High Vol"]
    elif n == 3:
        names = ["Low (Bull)", "Medium (Sideways)", "High (Bear)"]
    else:
        names = [f"Regime {i}" for i in range(n)]

    state_map = {state: names[i] for i, state in enumerate(vol_means.index)}
    df["regime"] = df["state"].map(state_map)
    return df, state_map