import numpy as np
from hmmlearn.hmm import GaussianHMM


def train_hmm(X: np.ndarray, n_states: int = 3, n_iter: int = 500, random_state: int = 42) -> GaussianHMM:

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)
    return model


def predict_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


MIN_STATE_OCCUPANCY = 0.01  # below this, a state has collapsed onto an outlier rather than modeling a regime


def state_occupancy(states: np.ndarray, n_states: int) -> np.ndarray:
    return np.bincount(states, minlength=n_states) / len(states)


def is_degenerate(states: np.ndarray, n_states: int, min_occupancy: float = MIN_STATE_OCCUPANCY) -> bool:
    return bool(state_occupancy(states, n_states).min() < min_occupancy)


def compute_metrics(model: GaussianHMM, X: np.ndarray) -> dict:
    
    n, n_features = X.shape
    n_states = model.n_components

    # Free parameters:
    #   transition matrix: n_states * (n_states - 1)   (rows sum to 1)
    #   start probs:       n_states - 1
    #   means:             n_states * n_features
    #   covariances:       depends on covariance_type
    k_trans   = n_states * (n_states - 1)
    k_start   = n_states - 1
    k_means   = n_states * n_features
    k_cov     = {
        "spherical": n_states,
        "diag":      n_states * n_features,
        "full":      n_states * n_features * (n_features + 1) // 2,
        "tied":      n_features * (n_features + 1) // 2,
    }[model.covariance_type]
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