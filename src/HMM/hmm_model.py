from hmmlearn.hmm import GaussianHMM
import numpy as np

def train_hmm(X, n_states=3):
    best_model = None
    best_score = -np.inf

    for seed in [0, 1, 2, 42, 100]:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",#independent training pa, palitan nalang "full" if gusto may correlation
            n_iter=1000,
            random_state=seed
        )
        
        model.fit(X)
        score = model.score(X)  #log-likelihood
        
        if score > best_score:
            best_score = score
            best_model = model

    return best_model

def predict_states(model, X):
    return model.predict(X)
