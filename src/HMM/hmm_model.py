from hmmlearn.hmm import GaussianHMM

def train_hmm(X, n_states=3):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=42
    )
    model.fit(X)
    return model


def predict_states(model, X):
    return model.predict(X)